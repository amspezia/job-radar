import pytest
from pydantic import BaseModel

from job_radar.adapters import tracing
from job_radar.config import settings


class _Schema(BaseModel):
    value: str
    secret: str


@pytest.fixture(autouse=True)
def _reset_capture_flag():
    original = settings.langfuse_capture_full_payload
    yield
    settings.langfuse_capture_full_payload = original


def test_trace_input_text_redacted_by_default() -> None:
    settings.langfuse_capture_full_payload = False
    text = "some prompt with CV text in it"
    result = tracing.trace_input_text(text)
    assert result == {"chars": len(text)}
    assert "prompt" not in str(result)


def test_trace_input_text_full_capture_when_enabled() -> None:
    settings.langfuse_capture_full_payload = True
    result = tracing.trace_input_text("some prompt")
    assert result == {"text": "some prompt"}


def test_trace_output_model_redacted_by_default() -> None:
    settings.langfuse_capture_full_payload = False
    result = tracing.trace_output_model(_Schema(value="ok", secret="do-not-leak"))
    assert result == {"schema": "_Schema"}
    assert "do-not-leak" not in str(result)


def test_trace_output_model_full_capture_when_enabled() -> None:
    settings.langfuse_capture_full_payload = True
    result = tracing.trace_output_model(_Schema(value="ok", secret="visible-when-opted-in"))
    assert result == {"value": "ok", "secret": "visible-when-opted-in"}


def test_trace_output_vector_never_includes_raw_values() -> None:
    settings.langfuse_capture_full_payload = False
    assert tracing.trace_output_vector([0.1, 0.2, 0.3]) == {"dims": 3}

    settings.langfuse_capture_full_payload = True
    # Vector output doesn't branch on the flag — length is never sensitive.
    assert tracing.trace_output_vector([0.1, 0.2, 0.3]) == {"dims": 3}
