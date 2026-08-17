import logging
from typing import Protocol

import httpx
from pydantic import BaseModel

from job_radar.adapters.retry import with_retry
from job_radar.config import settings

logger = logging.getLogger(__name__)

# Prompt + output must both fit here. Measured over 300 random jobs, fit prompts
# run p50 2719 / p90 2968 / max 4404 tokens; with a typical ~1100-token judgment,
# 10% of jobs overflowed the previous 4096 and the longest prompt did not fit at
# all (Ollama then silently drops the *front* of the prompt — the profile and CV).
# 8192 clears the whole corpus with room for a 1500-token answer.
_NUM_CTX = 8192


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
            "options": {"temperature": 0, "num_ctx": _NUM_CTX},
        }

        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(url=f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp

        resp = await with_retry(_call, label=f"generate({schema.__name__})")
        body = resp.json()
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
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url=f"{settings.ollama_base_url}/api/embed", json=payload)
            resp.raise_for_status()
            return resp

        resp = await with_retry(_call, label="embed")
        return resp.json()["embeddings"][0]


def get_provider() -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
