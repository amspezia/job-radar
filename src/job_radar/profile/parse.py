import logging

from job_radar.adapters.generation import generate
from job_radar.profile.schema import StructuredProfile

logger = logging.getLogger(__name__)

_PROMPT = """
You are extracting structured facts from a candidate's CV.
Extract ONLY information explicitly present in the CV text below.
Do NOT invent skills, roles, employers, or experience that are not stated.
If a field is not present, use null (or an empty list).

For each work_history entry capture role, company, start, end, the duration in
years, and the notable achievements/responsibilities as `highlights` (quote them
from the CV).

For `domains`, name the business/problem areas the candidate has worked in
(e.g. "fintech", "proptech", "telecom", "LLM/agent infrastructure"). These may
be inferred from the employers and project descriptions even when not stated as
a single word — this is the one field where reasonable inference is allowed.

CV:
{cv_text}
"""


async def parse_cv(cv_text: str) -> StructuredProfile:
    # Log only the size, never the CV content (PII).
    logger.debug("Parsing CV (%d chars) into a structured profile", len(cv_text))
    prompt = _PROMPT.format(cv_text=cv_text)
    return await generate(prompt, StructuredProfile)
