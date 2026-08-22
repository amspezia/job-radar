from typing import Literal

from langfuse import get_client

from job_radar.adapters.providers import get_provider
from job_radar.adapters.tracing import trace_input_text, trace_output_vector
from job_radar.config import settings

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
    # Same "generation"-level treatment as generate() (as_type="embedding" is
    # Langfuse's dedicated type for this — still cost/token-tracked, not
    # demoted to a plain span) and the same provider-agnostic seam: OllamaProvider
    # may enrich the current observation with usage via update_current_generation().
    client = get_client()
    with client.start_as_current_observation(
        name="embed",
        as_type="embedding",
        model=settings.embedding_model,
        input=trace_input_text(prefixed),
        metadata={"task": task},
    ) as obs:
        vector = await get_provider().embed(prefixed)
        obs.update(output=trace_output_vector(vector))
    return vector
