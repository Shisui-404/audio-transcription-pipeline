from typing import Literal

from pydantic import BaseModel, ConfigDict


class Segment(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    status: Literal['success'] = 'success'
    language: str | None
    duration_seconds: float
    text: str
    segments: list[Segment]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class PipelineError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
