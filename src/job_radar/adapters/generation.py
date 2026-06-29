import logging

import httpx
from pydantic import BaseModel

from job_radar.config import settings

logger = logging.getLogger(__name__)


async def generate[ModelT: BaseModel](prompt: str, schema: type[ModelT]) -> ModelT:
    # Log the call metadata only — never the prompt, which may carry PII.
    logger.debug(
        "generate: model=%s schema=%s prompt_chars=%d",
        settings.generation_model,
        schema.__name__,
        len(prompt),
    )
    payload = {
        "model": settings.generation_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema.model_json_schema(),
        "options": {"temperature": 0},
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url=f"{settings.ollama_base_url}/api/chat", json=payload)
    resp.raise_for_status()

    content = resp.json()["message"]["content"]
    return schema.model_validate_json(content)
