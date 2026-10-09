from pathlib import Path
from typing import BinaryIO

from app.schemas import PipelineError


def format_size(size: int) -> str:
    return f'{size:,} bytes ({size / (1024 * 1024):g} MiB)'


def save_upload(source: BinaryIO, destination: Path, limit: int) -> None:
    size = 0
    with destination.open('wb') as output:
        while chunk := source.read(min(1024 * 1024, limit + 1 - size)):
            size += len(chunk)
            if size > limit:
                raise PipelineError(413, 'file_too_large',
                                    f'The file exceeds the maximum allowed size of {format_size(limit)}.')
            output.write(chunk)
    if not size:
        raise PipelineError(400, 'empty_file', 'The file is empty.')
