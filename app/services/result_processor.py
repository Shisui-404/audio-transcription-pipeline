import math

from app.schemas import PipelineError, Segment, Transcript


def assemble(raw_segments, duration: float, language: str | None) -> Transcript:
    segments = []
    previous_end = -math.inf
    for raw in raw_segments:
        start, end = raw.start, raw.end
        if (not math.isfinite(start) or not math.isfinite(end)
                or start > end or start < -0.05 or end > duration + 0.05
                or start < previous_end):
            raise PipelineError(500, 'invalid_engine_result', 'The engine returned invalid timestamps.')
        start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
        previous_end = end
        if text := raw.text.strip():
            segments.append(Segment(start=start, end=end, text=text))
    return Transcript(language=language, duration_seconds=duration,
                      text=' '.join(s.text for s in segments), segments=segments)
