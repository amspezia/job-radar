"""Redaction helpers shared by generation.py and embeddings.py's Langfuse spans.

Safe by default: raw prompt/response text never reaches a trace unless
`settings.langfuse_capture_full_payload` is explicitly set. See
docs/plans/phase-c/02-core-instrumentation-and-redaction.md's redaction table —
CV text, profile PII, and FitJudgment's Evidence.quote fields (verbatim CV/posting
excerpts by design, per fit/schema.py) all flow through prompts and structured
outputs, so the default has to be "summary only," not "everything except a few
fields we remembered to strip."
"""

from pydantic import BaseModel

from job_radar.config import settings


def trace_input_text(text: str) -> dict:
    """Redacted-by-default representation of a prompt or text-to-embed string."""
    if settings.langfuse_capture_full_payload:
        return {"text": text}
    return {"chars": len(text)}


def trace_output_model(result: BaseModel) -> dict:
    """Redacted-by-default representation of a schema-constrained generation result.

    Full capture intentionally includes everything in the model, including any
    Evidence.quote-style verbatim fields — that's the understood tradeoff of
    turning the flag on at all (see config.py's langfuse_capture_full_payload
    docstring), not a gap to close with per-field allowlisting here.
    """
    if settings.langfuse_capture_full_payload:
        return result.model_dump(mode="json")
    return {"schema": type(result).__name__}


def trace_output_vector(vector: list[float]) -> dict:
    """Embedding output is a vector, not text — its length is never sensitive,
    so this doesn't need to branch on langfuse_capture_full_payload at all.
    """
    return {"dims": len(vector)}
