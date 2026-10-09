from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    model_size: Literal['small', 'base'] = 'small'
    model_revision: str | None = None
    model_cache: str = '.cache/models'
    model_local_files_only: bool = False
    device: Literal['cpu', 'cuda'] = 'cpu'
    compute_type: Literal['int8', 'float16', 'float32'] = 'int8'
    cpu_threads: int = Field(default=4, ge=1)
    beam_size: int = Field(default=5, ge=1)
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    vad_min_silence_duration_ms: int = Field(default=500, ge=0)
    vad_speech_pad_ms: int = Field(default=400, ge=0)
    max_upload_bytes: int = Field(default=52428800, gt=0)
    max_audio_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    preprocess_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_concurrent_transcriptions: Literal[1] = 1
    ffmpeg_path: str = 'ffmpeg'
    ffprobe_path: str = 'ffprobe'
    temp_dir: str | None = None
