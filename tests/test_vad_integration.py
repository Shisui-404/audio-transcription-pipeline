from pathlib import Path
import subprocess
import wave

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def real_client():
    with TestClient(create_app(Settings())) as client:
        assert client.get('/ready').status_code == 200
        yield client


@pytest.mark.parametrize('name', ['speech.wav', 'timing.wav', 'timing.mp3', 'quiet.wav'])
def test_real_speech(real_client, name, synthetic_audio):
    response = real_client.post('/transcribe', files={'file': ('incorrect.extension', (synthetic_audio / name).read_bytes())})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['text'] and data['segments'] and data['language'] == 'en'
    previous = 0
    for segment in data['segments']:
        assert previous <= segment['start'] <= segment['end'] <= data['duration_seconds']
        previous = segment['end']
    if name.startswith('timing'):
        with wave.open(str(synthetic_audio / 'mono.wav')) as audio:
            speech_duration = audio.getnframes() / audio.getframerate()
        assert data['segments'][0]['start'] >= 2.5
        assert any(s['start'] >= 3 + speech_duration + 3.5 for s in data['segments'])


@pytest.mark.parametrize('name', ['silence.wav', 'noise.wav', 'tone.wav'])
def test_real_no_speech(real_client, name, synthetic_audio):
    response = real_client.post('/transcribe', files={'file': (name, (synthetic_audio / name).read_bytes())})
    assert response.status_code == 200
    assert response.json()['text'] == '' and response.json()['segments'] == []
    assert response.json()['language'] is None


def test_real_corrupt(real_client):
    assert real_client.post('/transcribe', files={'file': ('fake.wav', b'not audio')}).status_code == 400


def test_real_unsupported_content(real_client, tmp_path, synthetic_audio):
    flac = tmp_path / 'audio.flac'
    subprocess.run([Settings().ffmpeg_path, '-v', 'error', '-i', str(synthetic_audio / 'mono.wav'), str(flac)], check=True)
    response = real_client.post('/transcribe', files={'file': ('disguised.wav', flac.read_bytes())})
    assert response.status_code == 415


def test_real_stereo_resampling(real_client, tmp_path, synthetic_audio):
    stereo = tmp_path / 'stereo.wav'
    subprocess.run([Settings().ffmpeg_path, '-v', 'error', '-i', str(synthetic_audio / 'mono.wav'),
                    '-ac', '2', '-ar', '44100', str(stereo)], check=True)
    response = real_client.post('/transcribe', files={'file': ('stereo.wav', stereo.read_bytes())})
    assert response.status_code == 200 and response.json()['text']
    with wave.open(str(stereo)) as audio:
        assert abs(response.json()['duration_seconds'] - audio.getnframes() / audio.getframerate()) < 0.01
