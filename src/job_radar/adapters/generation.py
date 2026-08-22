import logging

from langfuse import get_client
from pydantic import BaseModel

from job_radar.adapters.providers import TruncatedGeneration as TruncatedGeneration
from job_radar.adapters.providers import get_provider
from job_radar.adapters.tracing import trace_input_text, trace_output_model
from job_radar.config import settings

logger = logging.getLogger(__name__)


async def generate[ModelT: BaseModel](
    prompt: str,
    schema: type[ModelT],
    *,
    model: str | None = None,
) -> ModelT:
    effective_model = model or settings.generation_model
    # Log the call metadata only — never the prompt, which may carry PII.
    logger.debug(
        "generate: model=%s schema=%s prompt_chars=%d",
        effective_model,
        schema.__name__,
        len(prompt),
    )
    # The provider-agnostic seam: every LLMProvider implementation (today just
    # OllamaProvider, later a Phase E paid-API provider) flows through this one
    # observation, so tracing lives here rather than inside each provider — a
    # future provider gets identical tracing for free. Providers may enrich the
    # *current* observation (usage, retry count) via update_current_generation();
    # see adapters/providers.py.
    client = get_client()
    with client.start_as_current_observation(
        name=f"generate.{schema.__name__}",
        as_type="generation",
        model=effective_model,
        input=trace_input_text(prompt),
    ) as obs:
        result = await get_provider().generate(prompt, schema, model=effective_model)
        obs.update(output=trace_output_model(result))
    return result
