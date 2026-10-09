import io
import json
import subprocess
import wave

import pytest

from app.schemas import PipelineError
from app.services import audio_processor as audio
from app.utils.file_validator import save_upload
from conftest import wav_bytes


@pytest.mark.parametrize('content,limit,code', [(b'', 10, 'empty_file'), (b'12345', 4, 'file_too_large')])
def test_upload_errors(tmp_path, content, limit, code):
    with pytest.raises(PipelineError) as exc:
        save_upload(io.BytesIO(content), tmp_path / 'upload', limit)
    assert exc.value.code == code
    if code == 'file_too_large':
        assert f'{limit:,} bytes' in exc.value.message
        assert 'MiB' in exc.value.message


def test_upload_exact_limit(tmp_path):
    save_upload(io.BytesIO(b'1234'), tmp_path / 'upload', 4)
    assert (tmp_path / 'upload').read_bytes() == b'1234'


@pytest.mark.parametrize('probe,status', [({'streams': []}, 400),
    ({'streams': [{}], 'format': {'format_name': 'flac'}}, 415),
    ({'streams': [{}], 'format': {'format_name': 'wav', 'duration': '2000'}}, 413)])
def test_probe_rejections(monkeypatch, tmp_path, settings, probe, status):
    monkeypatch.setattr(audio, 'run', lambda *args: json.dumps(probe).encode())
    with pytest.raises(PipelineError) as exc:
        audio.normalize(tmp_path / 'input', tmp_path / 'output', settings)
    assert exc.value.status == status


def test_decoded_duration_enforced(monkeypatch, tmp_path, settings):
    settings.max_audio_seconds = 1
    def fake_run(command, timeout):
        if command[0] == settings.ffprobe_path:
            return b'{"streams":[{}],"format":{"format_name":"wav","duration":"0.1"}}'
        (tmp_path / 'output').write_bytes(wav_bytes(2, 16000, 1))
        assert '-t' in command and '-fs' in command
        return b''
    monkeypatch.setattr(audio, 'run', fake_run)
    with pytest.raises(PipelineError) as exc:
        audio.normalize(tmp_path / 'input', tmp_path / 'output', settings)
    assert exc.value.code == 'audio_too_long'


def test_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ffmpeg', 1)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(PipelineError) as exc:
        audio.run(['ffmpeg'], 1)
    assert exc.value.status == 504
