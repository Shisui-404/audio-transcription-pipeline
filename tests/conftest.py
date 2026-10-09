import io
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import wave

import pytest

from app.config import Settings


@pytest.fixture(scope='session')
def synthetic_audio(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_audio')
    build_synthetic_audio(root)
    return root


@pytest.fixture(scope='session')
def human_audio(tmp_path_factory):
    root = tmp_path_factory.mktemp('human_audio')
    return root, build_human_audio(root)['fixtures']


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, temp_dir=str(tmp_path))


def wav_bytes(seconds=1, rate=44100, channels=2):
    result = io.BytesIO()
    with wave.open(result, 'wb') as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b'\0\0' * int(seconds * rate) * channels)
    return result.getvalue()


def write(path, samples):
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.asarray(samples, dtype='<i2').tobytes())


def build_synthetic_audio(root):
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile('tests/fixtures/speech.wav', root / 'speech.wav')
    ffmpeg = Settings().ffmpeg_path
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(root / 'speech.wav'),
                    '-ac', '1', '-ar', '16000', str(root / 'mono.wav')], check=True)
    with wave.open(str(root / 'mono.wav'), 'rb') as source:
        speech = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2')
    write(root / 'timing.wav', np.concatenate([np.zeros(48000), speech, np.zeros(64000), speech, np.zeros(16000)]))
    write(root / 'silence.wav', np.zeros(80000))
    write(root / 'quiet.wav', speech.astype(float) * 0.08)
    write(root / 'noise.wav', np.random.default_rng(42).normal(0, 300, 80000))
    write(root / 'tone.wav', 2000 * np.sin(np.arange(80000) * 2 * np.pi * 440 / 16000))
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(root / 'timing.wav'),
                    '-c:a', 'libmp3lame', str(root / 'timing.mp3')], check=True)


ROOT = Path('sample_audio')


def build_human_audio(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    fixtures = []
    for sample in manifest['samples']:
        for extension in ('wav', 'mp3'):
            name = f"{sample['id']}.{extension}"
            subprocess.run([Settings().ffmpeg_path, '-v', 'error', '-y', '-i',
                            str(ROOT / sample['file']), '-ac', '1', '-ar', '16000',
                            str(root / name)], check=True)
            fixtures.append({'file': name, 'reference_text': sample['reference_text'],
                             'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest(),
                             'changes': f"Converted from canonical {sample['file']}; no trimming or padding."})
    first = manifest['samples'][0]
    with wave.open(str(root / f"{first['id']}.wav")) as source:
        frames = source.readframes(source.getnframes())
        duration = source.getnframes() / source.getframerate()
    with wave.open(str(root / 'timing.wav'), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\0' * 96000 + frames + b'\0' * 128000 + frames + b'\0' * 32000)
    fixtures.append({'file': 'timing.wav', 'reference_text': f"{first['reference_text']} {first['reference_text']}",
                     'sha256': hashlib.sha256((root / 'timing.wav').read_bytes()).hexdigest(),
                     'changes': 'First human clip repeated, with 3 s leading, 4 s internal and 1 s trailing silence.',
                     'speech_intervals_seconds': [[3, 3 + duration], [7 + duration, 7 + 2 * duration]]})
    return {**manifest, 'fixtures': fixtures}
