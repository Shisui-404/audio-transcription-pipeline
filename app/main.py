import asyncio
import logging
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from starlette.datastructures import UploadFile
from starlette.responses import JSONResponse

from app.config import Settings
from app.schemas import ErrorResponse, PipelineError, Transcript
from app.services.audio_processor import normalize
from app.services.transcriber import LANGUAGES, Transcriber
from app.utils.file_validator import format_size, save_upload

logger = logging.getLogger(__name__)


def create_app(settings=None, engine_factory=Transcriber):
    settings = settings or Settings()
    slot = threading.BoundedSemaphore(1)
    tasks = set()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        try:
            for executable in (settings.ffmpeg_path, settings.ffprobe_path):
                if not shutil.which(executable):
                    raise RuntimeError(f'Missing executable: {executable}')
            app.state.engine = await asyncio.to_thread(engine_factory, settings)
        except Exception:
            logger.exception('Transcription startup failed')
        yield
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        app.state.engine = None

    app = FastAPI(title='Local audio transcription', lifespan=lifespan)

    @app.exception_handler(PipelineError)
    async def pipeline_error(request, exc):
        return JSONResponse(status_code=exc.status,
                            content={'error': {'code': exc.code, 'message': exc.message}},
                            headers={'Retry-After': '5'} if exc.code == 'busy' else None)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(status_code=exc.status_code,
                            content={'error': {'code': 'invalid_request', 'message': str(exc.detail)}})

    @app.get('/health')
    async def health():
        return {'status': 'alive'}

    @app.get('/ready')
    async def ready():
        if app.state.engine is None:
            raise PipelineError(503, 'unavailable', 'The transcription service is not ready.')
        return {'status': 'ready'}

    def process(upload, language):
        with tempfile.TemporaryDirectory(dir=settings.temp_dir, prefix='transcribe-') as directory:
            source, output = Path(directory) / 'upload', Path(directory) / 'normalized.wav'
            save_upload(upload.file, source, settings.max_upload_bytes)
            duration = normalize(source, output, settings)
            return app.state.engine.transcribe(output, duration, language)

    async def execute(request):
        # This shielded task owns parser files and admission until native work finishes.
        try:
            async with request.form(max_files=1, max_fields=1, max_part_size=1024) as form:
                upload, language = form.get('file'), form.get('language')
                # Swagger UI can send an empty optional field instead of omitting it.
                if isinstance(language, str):
                    language = language.strip() or None
                if (not isinstance(upload, UploadFile) or set(form) - {'file', 'language'}
                        or len(form.getlist('file')) != 1 or len(form.getlist('language')) > 1
                        or (language is not None and (not isinstance(language, str) or language not in LANGUAGES))):
                    raise PipelineError(422, 'invalid_parameters', 'Supply one audio file and an optional supported language code.')
                return await asyncio.to_thread(process, upload, language)
        except (PipelineError, HTTPException):
            raise
        except Exception as exc:
            logger.error('Transcription failed (%s)', type(exc).__name__)
            raise PipelineError(500, 'inference_failed', 'Audio processing failed.') from exc
        finally:
            slot.release()

    @app.post('/transcribe', response_model=Transcript,
              responses={s: {'model': ErrorResponse} for s in (400, 413, 415, 422, 500, 503, 504)},
              openapi_extra={'requestBody': {'required': True, 'content': {'multipart/form-data': {
                  'schema': {'type': 'object', 'required': ['file'], 'properties': {
                      'file': {'type': 'string', 'format': 'binary'},
                      'language': {'type': 'string', 'description': 'Optional Whisper language code (e.g. en). Leave blank for automatic detection.'}}}}}}})
    async def transcribe(request: Request):
        if app.state.engine is None:
            raise PipelineError(503, 'unavailable', 'The transcription service is not ready.')
        if not slot.acquire(blocking=False):
            raise PipelineError(503, 'busy', 'Processing capacity is occupied.')
        task = asyncio.create_task(execute(request))
        tasks.add(task)
        def finished(done):
            tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Retrieve failures even when the client has disconnected.
        task.add_done_callback(finished)
        return await asyncio.shield(task)

    return BodyLimitMiddleware(app, settings.max_upload_bytes + 65536, settings.max_upload_bytes)


class BodyLimitMiddleware:
    """Bound streamed multipart bodies before the parser can fill temporary storage."""
    def __init__(self, app, limit, file_limit):
        self.app, self.limit, self.file_limit = app, limit, file_limit

    def __getattr__(self, name):
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        size = 0
        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get('body', b''))
            if size > self.limit:
                raise PipelineError(
                    413, 'file_too_large',
                    f'The request exceeds the maximum upload body size of {format_size(self.limit)}, '
                    f'including multipart overhead. Maximum file size: {format_size(self.file_limit)}.',
                )
            return message
        await self.app(scope, bounded_receive, send)


app = create_app()
