import json
import math
import subprocess
import wave
from pathlib import Path

from app.config import Settings
from app.schemas import PipelineError


def run(command: list[str], timeout: float) -> bytes:
    # Only ffprobe's fixed selected fields reach stdout; diagnostics are discarded.
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=timeout, check=True).stdout
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(504, 'preprocessing_timeout', 'Audio preprocessing timed out.') from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(400, 'invalid_audio', 'The audio could not be decoded.') from exc
    except FileNotFoundError as exc:
        raise PipelineError(503, 'unavailable', 'A required audio executable is unavailable.') from exc


def normalize(source: Path, output: Path, settings: Settings) -> float:
    timeout = settings.preprocess_timeout_seconds
    probe = json.loads(run([
        settings.ffprobe_path, '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-select_streams', 'a:0', '-show_entries',
        'format=format_name,duration:stream=codec_name', '-of', 'json', str(source),
    ], timeout))
    if not probe.get('streams'):
        raise PipelineError(400, 'invalid_audio', 'No decodable audio stream was found.')
    formats = set(probe.get('format', {}).get('format_name', '').split(','))
    if not formats.intersection({'wav', 'mp3'}):
        raise PipelineError(415, 'unsupported_format', 'Only WAV and MP3 content is supported.')
    try:
        duration = float(probe['format'].get('duration', 'nan'))
    except ValueError:
        duration = math.nan
    if math.isfinite(duration) and duration > settings.max_audio_seconds + 0.05:
        raise PipelineError(413, 'audio_too_long', 'The recording exceeds the duration limit.')
    ceiling = settings.max_audio_seconds + 1
    run([
        settings.ffmpeg_path, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
        '-protocol_whitelist', 'file,pipe', '-i', str(source), '-map', '0:a:0',
        '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
        '-t', str(ceiling), '-fs', str(math.ceil(ceiling * 32000) + 65536),
        '-y', str(output),
    ], timeout)
    with wave.open(str(output), 'rb') as audio:
        duration = audio.getnframes() / audio.getframerate()
    if duration > settings.max_audio_seconds:
        raise PipelineError(413, 'audio_too_long', 'The decoded recording exceeds the duration limit.')
    if duration <= 0:
        raise PipelineError(400, 'invalid_audio', 'The audio contains no samples.')
    return duration
