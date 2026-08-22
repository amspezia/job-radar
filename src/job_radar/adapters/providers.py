import logging
from typing import Protocol

import httpx
from langfuse import get_client
from pydantic import BaseModel

from job_radar.adapters.retry import with_retry
from job_radar.config import settings

logger = logging.getLogger(__name__)

# A flat timeout=N applies to connect/read/write/pool alike, so a single stuck
# request (Ollama accepted the connection but never responded — observed in
# practice under concurrent load) blocked its with_retry attempt, and the
# semaphore slot it held in fit/pipeline.py, for the *entire* N seconds before
# even raising to retry. Splitting connect/write/pool (should be near-instant
# against a local Ollama) from read (the only leg that legitimately takes
# a while) lets a genuine hang fail fast and free its slot for other jobs,
# without needing every request to actually be fast.
# The read budget must exceed the slowest *legitimate* completion, or it stops
# being a hang detector and becomes a throughput bug. Measured 2026-08-21: at 12
# concurrent slots each stream decodes ~8.8 tok/s, so the previous read=120 cut
# off at ~1,050 output tokens — and 11.9% of FitJudgments are longer than that
# (p90 1207, p95 1820, max 2770). Those requests were aborted mid-decode, retried
# by with_retry, and — because temperature is 0 — regenerated the *identical*
# over-long answer and timed out again, until all three attempts were spent.
# Net effect: ~33% of decode time thrown away and 6.2% of jobs returning no
# judgment at all. _NUM_PREDICT below now bounds a run away answer, so this only
# has to cover the worst bounded case (3072 tok ÷ 8.8 tok/s ≈ 350 s) with margin.
_GENERATE_TIMEOUT = httpx.Timeout(connect=10.0, read=420.0, write=10.0, pool=10.0)
_EMBED_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

# Prompt + output must both fit here. Measured over 300 random jobs, fit prompts
# run p50 2719 / p90 2968 / max 4404 tokens; with a typical ~1100-token judgment,
# 10% of jobs overflowed the previous 4096 and the longest prompt did not fit at
# all (Ollama then silently drops the *front* of the prompt — the profile and CV).
# 8192 clears the whole corpus with room for a 1500-token answer.
_NUM_CTX = 8192

# Hard ceiling on output tokens. Ollama's default is unbounded-up-to-num_ctx, so a
# model that starts enumerating requirements without stopping can decode ~5,700
# tokens against a typical prompt — over ten minutes of GPU time for an answer
# that is pathological rather than thorough. 3072 sits above the longest real
# judgment observed (2,770 tok over 173 traced calls) and below what the context
# leaves free even for the 4,404-token worst-case prompt. Exceeding it sets
# done_reason="length", which surfaces as TruncatedGeneration — deliberately not
# a transient error, so with_retry does not re-run a generation that would only
# overflow again.
_NUM_PREDICT = 3072

# Loading qwen2.5 at num_ctx 8192 takes far longer than a generate call's 120s read
# budget, and Ollama unloads after ~5 minutes idle — so the first batch after a pause
# hits a cold model. Every concurrent request then waits on that one load and times
# out together, and with_retry's three attempts can expire before it finishes
# (measured: 2 of 6 jobs lost that way). warm() pays the load once, up front,
# with a budget sized for it instead of for a steady-state request.
_WARM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)


class TruncatedGeneration(Exception):
    """The model hit the token ceiling mid-answer, so its output is incomplete.

    Worth its own type because schema-constrained decoding cannot emit malformed
    JSON — if parsing fails, the cause is almost always truncation, and saying so
    beats a downstream pydantic error about an unexpected EOF.
    """


class LLMProvider(Protocol):
    """A backend that can turn a prompt into structured output and text into a vector.

    Ollama is the only implementation today (see OllamaProvider below). A second
    provider is a new class satisfying this Protocol plus a branch in get_provider() —
    the generation.py/embeddings.py dispatchers and every caller stay unchanged.
    """

    async def generate[ModelT: BaseModel](
        self, prompt: str, schema: type[ModelT], *, model: str
    ) -> ModelT: ...

    async def embed(self, text: str) -> list[float]:
        """Embed already-prepared text (e.g. with any task prefix applied)."""
        ...

    async def warm(self, model: str) -> None:
        """Ensure `model` is ready to serve, so a batch's first burst isn't the load.

        A no-op for hosted providers, which have no load step. Callers treat this
        as best-effort: a failure here must not abort the work that follows.
        """
        ...


class OllamaProvider:
    """Talks to a local Ollama instance over its /api/chat and /api/embed endpoints."""

    async def generate[ModelT: BaseModel](
        self, prompt: str, schema: type[ModelT], *, model: str
    ) -> ModelT:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema.model_json_schema(),
            # Off, not merely unset. A thinking-capable model (e.g. qwen3:4b) left
            # to its default reasons inside <think> before answering; measured
            # 2026-08-21, that reasoning ran past num_predict on every one of 3
            # calls, so the response was cut off mid-thought and NEVER reached the
            # JSON answer — a 100% FitJudgment parse failure, not a slowdown. A
            # model without a thinking mode (qwen2.5) ignores this field.
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": _NUM_CTX,
                "num_predict": _NUM_PREDICT,
            },
        }

        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT) as client:
                resp = await client.post(url=f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp

        attempts = 0

        def _record_attempt(n: int) -> None:
            nonlocal attempts
            attempts = n

        resp = await with_retry(
            _call, label=f"generate({schema.__name__})", on_attempt=_record_attempt
        )
        body = resp.json()
        # Enrich the *current* observation (created by generation.py's dispatcher,
        # not a new span here — see docs/plans/phase-c/02-core-instrumentation-and-
        # redaction.md's "Files touched" for why) — before the truncation check, so a
        # truncated call's real token usage still lands in the trace.
        get_client().update_current_generation(
            usage_details={
                "input": body.get("prompt_eval_count", 0),
                "output": body.get("eval_count", 0),
            },
            metadata={"retry_attempts": attempts},
        )
        if body.get("done_reason") == "length":
            raise TruncatedGeneration(
                f"{model} hit the {_NUM_CTX}-token ceiling "
                f"({body.get('prompt_eval_count')} prompt + {body.get('eval_count')} output); "
                "the answer is incomplete"
            )
        return schema.model_validate_json(body["message"]["content"])

    async def embed(self, text: str) -> list[float]:
        payload = {
            "model": settings.embedding_model,
            "input": text,
            # nomic-embed-text supports long context via rotary scaling (up to
            # 8192 tokens). Without raising num_ctx, Ollama's default (~2048) silently
            # truncates long job descriptions, dropping the tech-stack list at the end.
            "options": {"num_ctx": 8192},
        }

        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
                resp = await client.post(url=f"{settings.ollama_base_url}/api/embed", json=payload)
            resp.raise_for_status()
            return resp

        attempts = 0

        def _record_attempt(n: int) -> None:
            nonlocal attempts
            attempts = n

        resp = await with_retry(_call, label="embed", on_attempt=_record_attempt)
        body = resp.json()
        # Ollama's /api/embed exposes prompt_eval_count (input tokens processed) but,
        # confirmed against a real response, no eval_count — there's no generative
        # "output" token count for an embedding call, unlike /api/chat.
        get_client().update_current_generation(
            usage_details={"input": body.get("prompt_eval_count", 0)},
            metadata={"retry_attempts": attempts},
        )
        return body["embeddings"][0]

    async def warm(self, model: str) -> None:
        """Load `model` into memory via Ollama's empty-prompt preload.

        Deliberately outside with_retry and its own Langfuse observation: this
        generates no tokens, so there is nothing to trace, and a failure is not
        worth retrying — the generate calls that follow carry their own retries
        and will simply pay the load themselves.
        """
        async with httpx.AsyncClient(timeout=_WARM_TIMEOUT) as client:
            resp = await client.post(
                url=f"{settings.ollama_base_url}/api/generate", json={"model": model}
            )
        resp.raise_for_status()


def get_provider() -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
