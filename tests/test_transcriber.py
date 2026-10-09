from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import PipelineError
from app.services.result_processor import assemble
from app.services.transcriber import Transcriber


def segment(start=1, end=2, text=' hello '):
    return NS(start=start, end=end, text=text)


def test_adapter_vad_and_consumption(settings):
    engine = Transcriber.__new__(Transcriber)
    engine.settings = settings
    consumed = []
    def generate():
        consumed.append(True)
        yield segment()
    engine.model = Mock()
    engine.model.transcribe.return_value = (generate(), NS(language='en', duration_after_vad=1))
    result = engine.transcribe('test.wav', 4)
    assert consumed and result.text == 'hello' and result.segments[0].start == 1
    kwargs = engine.model.transcribe.call_args.kwargs
    assert kwargs['vad_filter'] is True
    assert kwargs['vad_parameters'] == {'threshold': 0.5, 'min_silence_duration_ms': 500, 'speech_pad_ms': 400}
    assert kwargs['word_timestamps'] is False
    assert kwargs['beam_size'] == 5 and kwargs['task'] == 'transcribe'


@pytest.mark.parametrize('retained', [0, 1])
@pytest.mark.parametrize('language', [None, 'en'])
def test_no_speech(settings, retained, language):
    engine = Transcriber.__new__(Transcriber)
    engine.settings = settings
    engine.model = Mock()
    engine.model.transcribe.return_value = (iter([]), NS(language='en', duration_after_vad=retained))
    result = engine.transcribe('test.wav', 4, language)
    assert result.language == language and result.segments == [] and result.text == ''


def test_delayed_iterator_failure(settings):
    engine = Transcriber.__new__(Transcriber)
    engine.settings = settings
    engine.model = Mock()
    def broken():
        yield segment()
        raise RuntimeError('delayed inference failure')
    engine.model.transcribe.return_value = (broken(), NS(language='en', duration_after_vad=1))
    with pytest.raises(RuntimeError, match='delayed'):
        engine.transcribe('test.wav', 4)


@pytest.mark.parametrize('raw', [segment(float('nan')), segment(end=float('inf')),
                                       segment(-0.06), segment(2, 1), segment(1, 4.06)])
def test_invalid_timestamps(raw):
    with pytest.raises(PipelineError, match='timestamps'):
        assemble([raw], 4, 'en')


def test_order_and_clamping():
    result = assemble([segment(-0.04, 1), segment(2, 4.04, ' world ')], 4, 'en')
    assert result.text == 'hello world'
    assert result.segments[0].start == 0 and result.segments[-1].end == 4
    with pytest.raises(PipelineError):
        assemble([segment(2, 3), segment(1, 2)], 4, 'en')


@pytest.mark.parametrize('options', [{'vad_threshold': 0}, {'vad_threshold': 1},
                                   {'vad_speech_pad_ms': -1}, {'max_concurrent_transcriptions': 2},
                                   {'max_audio_seconds': float('inf')}])
def test_configuration_validation(options):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **options)
