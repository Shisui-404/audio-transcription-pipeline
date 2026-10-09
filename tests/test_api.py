import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.schemas import PipelineError, Transcript


@pytest.fixture
def application(monkeypatch, settings):
    monkeypatch.setattr(main.shutil, 'which', lambda x: x)
    monkeypatch.setattr(main, 'normalize', lambda *args: 1.0)
    engine = SimpleNamespace(transcribe=lambda *args: Transcript(
        language='en', duration_seconds=1, text='Hello', segments=[]))
    return main.create_app(settings, lambda s: engine), engine


def test_contract_and_parameters(application):
    app, _ = application
    with TestClient(app) as client:
        assert client.get('/health').status_code == client.get('/ready').status_code == 200
        result = client.post('/transcribe', files={'file': ('wrong.ext', b'abc')})
        assert result.status_code == 200 and result.json()['text'] == 'Hello'
        for data in ({}, {'language': 'invalid'}, {'vad_filter': 'false'}):
            response = client.post('/transcribe', data=data,
                                   files={'file': ('audio.wav', b'abc')} if data else None)
            assert response.status_code == 422 and 'error' in response.json()


@pytest.mark.parametrize('language,expected', [(None, None), ('', None), ('   ', None), ('en', 'en')])
def test_optional_language(application, language, expected):
    app, engine = application
    received = []
    def transcribe(path, duration, requested_language):
        received.append(requested_language)
        return Transcript(language=requested_language, duration_seconds=duration, text='', segments=[])
    engine.transcribe = transcribe
    with TestClient(app) as client:
        response = client.post('/transcribe', files={'file': ('speech.wav', b'audio')},
                               data={} if language is None else {'language': language})
    assert response.status_code == 200, response.text
    assert received == [expected]


def test_limits_and_cleanup(application, settings):
    app, _ = application
    settings.max_upload_bytes = 3
    with TestClient(app) as client:
        for payload, status in [(b'', 400), (b'abcd', 413), (b'abc', 200)]:
            response = client.post('/transcribe', files={'file': ('x', payload)})
            assert response.status_code == status
            if status == 413:
                assert '3 bytes' in response.json()['error']['message']
            assert list(Path(settings.temp_dir).iterdir()) == []


@pytest.mark.parametrize('failure', [RuntimeError('engine'), PipelineError(504, 'preprocessing_timeout', 'timeout')])
def test_failure_cleanup_and_slot(application, settings, failure):
    app, engine = application
    def fail(*args):
        assert Path(args[0]).parent.exists()
        raise failure
    engine.transcribe = fail
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post('/transcribe', files={'file': ('x', b'abc')})
            assert response.status_code == (504 if isinstance(failure, PipelineError) else 500)
            assert list(Path(settings.temp_dir).iterdir()) == []


def test_unavailable(monkeypatch, settings):
    monkeypatch.setattr(main.shutil, 'which', lambda x: None)
    with TestClient(main.create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        assert client.post('/transcribe').status_code == 503


def test_cancellation_retains_slot_and_files(application, settings):
    app, engine = application
    entered, release = threading.Event(), threading.Event()
    def transcribe(path, *args):
        entered.set()
        assert release.wait(10)
        assert path.parent.exists()
        return Transcript(language=None, duration_seconds=1, text='', segments=[])
    engine.transcribe = transcribe
    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                task = asyncio.create_task(client.post('/transcribe', files={'file': ('x', b'abc')}))
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert list(Path(settings.temp_dir).iterdir())
                    busy = await client.post('/transcribe', files={'file': ('x', b'abc')})
                    assert busy.status_code == 503 and busy.headers['retry-after'] == '5'
                    assert (await client.get('/health')).status_code == 200
                finally:
                    release.set()
        assert not list(Path(settings.temp_dir).iterdir())
    asyncio.run(scenario())


def test_streamed_body_limit(application):
    app, _ = application
    app.limit = 200
    async def scenario():
        async def body():
            yield b'--x\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\n\r\n'
            yield b'a' * 300
            yield b'\r\n--x--\r\n'
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/transcribe', content=body(), headers={'Content-Type': 'multipart/form-data; boundary=x'})
                assert response.status_code == 413
                assert response.json()['error']['code'] == 'file_too_large'
                message = response.json()['error']['message']
                assert '200 bytes' in message and 'including multipart overhead' in message
                assert 'Maximum file size: 52,428,800 bytes (50 MiB)' in message
    asyncio.run(scenario())
