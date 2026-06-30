import logging

from pydantic import BaseModel

from job_radar.adapters.generation import generate
from job_radar.config import settings

logger = logging.getLogger(__name__)

_TRUNCATE_AT = 4000  # chars — captures all meaningful content, skips tail boilerplate


class _Extraction(BaseModel):
    requirements: str
    responsibilities: str


_PROMPT = """\
Read the job posting below and extract two fields.

requirements: all skills, technologies, qualifications, and experience the role asks for
responsibilities: what the person will actually do day-to-day

Rules:
- Plain prose only — no bullet points, no markdown, no headers.
- Do not include company overview, benefits, perks, or hiring process text.
- If a section is absent or indistinguishable, return an empty string for that field.
- Return only what is stated in the posting; do not add or infer.

Posting:
{description}
"""


async def extract_fields(description: str) -> tuple[str, str]:
    """Return (requirements, responsibilities) extracted from a job description.

    Returns ("", "") on empty input or LLM failure so the ingest pipeline
    continues without structured fields rather than aborting the posting.
    """
    if not description.strip():
        return "", ""
    try:
        result = await generate(
            _PROMPT.format(description=description[:_TRUNCATE_AT]),
            _Extraction,
            model=settings.extraction_model,
        )
        return result.requirements, result.responsibilities
    except Exception:
        logger.warning("Field extraction failed; structured fields will be empty for this posting")
        return "", ""
