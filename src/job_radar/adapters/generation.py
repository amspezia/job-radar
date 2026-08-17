import logging

from pydantic import BaseModel

from job_radar.adapters.providers import TruncatedGeneration as TruncatedGeneration
from job_radar.adapters.providers import get_provider
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
    return await get_provider().generate(prompt, schema, model=effective_model)
