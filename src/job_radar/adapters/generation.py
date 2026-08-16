import logging

import httpx
from pydantic import BaseModel

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
    payload = {
        "model": effective_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": _NUM_CTX},
    }

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(url=f"{settings.ollama_base_url}/api/chat", json=payload)
    resp.raise_for_status()

    body = resp.json()
    if body.get("done_reason") == "length":
        raise TruncatedGeneration(
            f"{effective_model} hit the {_NUM_CTX}-token ceiling "
            f"({body.get('prompt_eval_count')} prompt + {body.get('eval_count')} output); "
            "the answer is incomplete"
        )
    return schema.model_validate_json(body["message"]["content"])
