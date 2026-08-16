from typing import Literal

import httpx

from job_radar.adapters.retry import with_retry
from job_radar.config import settings

# nomic-embed-text was contrastively trained with task-instruction prefixes.
# Omitting them places query and document vectors in a mismatched region of
# the space, degrading retrieval quality. The model card marks them required.
_PREFIX: dict[str, str] = {
    "query": "search_query",
    "document": "search_document",
}


async def embed(text: str, *, task: Literal["query", "document"]) -> list[float]:
    """Embed text with the correct nomic task prefix for asymmetric retrieval.

    Use task="query" for search-time text (query string, dense query).
    Use task="document" for indexed text (job postings, CV).
    """
    prefixed = f"{_PREFIX[task]}: {text}"
    payload = {
        "model": settings.embedding_model,
        "input": prefixed,
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
