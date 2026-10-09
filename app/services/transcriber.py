from faster_whisper import WhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES
from faster_whisper.utils import download_model

from app.services.result_processor import assemble

LANGUAGES = frozenset(_LANGUAGE_CODES)


class Transcriber:
    def __init__(self, settings):
        self.settings = settings
        self.model_path = download_model(
            settings.model_size, cache_dir=settings.model_cache,
            revision=settings.model_revision or (
                '536b0662742c02347bc0e980a01041f333bce120' if settings.model_size == 'small' else None
            ), local_files_only=settings.model_local_files_only,
        )
        self.model = WhisperModel(self.model_path, device=settings.device,
                                  compute_type=settings.compute_type, cpu_threads=settings.cpu_threads)

    def transcribe(self, path, duration, language=None):
        s = self.settings
        iterator, info = self.model.transcribe(
            str(path), language=language, task='transcribe', beam_size=s.beam_size,
            word_timestamps=False, vad_filter=True,
            vad_parameters={'threshold': s.vad_threshold,
                            'min_silence_duration_ms': s.vad_min_silence_duration_ms,
                            'speech_pad_ms': s.vad_speech_pad_ms},
        )
        segments = list(iterator)
        self.last_duration_after_vad = info.duration_after_vad
        if info.duration_after_vad == 0:
            segments = []
        result = assemble(segments, duration, language or info.language)
        if not result.text:
            result.language = language
        return result
