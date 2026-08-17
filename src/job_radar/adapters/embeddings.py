from typing import Literal

from job_radar.adapters.providers import get_provider

# nomic-embed-text was contrastively trained with task-instruction prefixes.
# Omitting them places query and document vectors in a mismatched region of
# the space, degrading retrieval quality. The model card marks them required.
#
# This is a property of the configured embedding model, not of the runtime
# serving it, so it lives here (the provider-agnostic dispatcher) rather than
# in the Ollama provider — a future non-nomic embedding model would set
# _PREFIX differently or drop it, independent of which provider serves it.
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
    return await get_provider().embed(prefixed)
