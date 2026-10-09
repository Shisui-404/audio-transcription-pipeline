"""Human-recorded LibriSpeech samples; no network access during test execution."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration
MANIFEST = json.loads(Path('sample_audio/manifest.json').read_text())
FILES = [f"{sample['id']}.{extension}" for sample in MANIFEST['samples'] for extension in ('wav', 'mp3')] + ['timing.wav']


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app(Settings())) as client:
        assert client.get('/ready').status_code == 200
        yield client


@pytest.mark.parametrize('name', FILES)
def test_human_speech(client, name, human_audio):
    root, fixtures = human_audio
    fixture = next(item for item in fixtures if item['file'] == name)
    path = root / name
    result = client.post('/transcribe', files={'file': (path.name, path.read_bytes())})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data['text'] and data['language'] == 'en' and data['segments']
    previous_end = 0
    for segment in data['segments']:
        assert previous_end <= segment['start'] <= segment['end'] <= data['duration_seconds']
        previous_end = segment['end']
    if intervals := fixture.get('speech_intervals_seconds'):
        assert data['segments'][0]['start'] >= intervals[0][0] - 0.5
        # Transcription segments may span a pause; test original timeline extent.
        # Transcript completeness is measured separately against the reference (WER).
        assert data['segments'][-1]['end'] >= intervals[1][0]
