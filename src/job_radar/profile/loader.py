import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.adapters.embeddings import embed
from job_radar.db.models import Profile
from job_radar.profile.extract import extract_text
from job_radar.profile.parse import parse_cv
from job_radar.retrieval.seniority import normalize_level

logger = logging.getLogger(__name__)


async def load_profile(session: AsyncSession, path: Path) -> Profile:
    """Parse a CV file into the single stored Profile (upsert), returning it.

    CV-derived fields are written on every load; preference fields
    (salary, remote, location rules, links) are not in a CV, so they're
    defaulted on first creation and preserved on later re-loads.
    """
    # Logs record only metadata (sizes, counts, seniority) — never the CV text,
    # name, or email, which are PII.
    logger.info("Loading profile from %s", path)

    text = extract_text(path)
    logger.info("Extracted %d characters of CV text", len(text))

    structured = await parse_cv(text)
    logger.info(
        "Parsed CV: seniority=%s, %d skills, %d work entries",
        structured.seniority,
        len(structured.tech_stack),
        len(structured.work_history),
    )

    embedding = await embed(text)
    logger.debug("Computed CV embedding (%d dims)", len(embedding))

    profile = (await session.execute(select(Profile))).scalars().first()
    created = profile is None
    if profile is None:
        profile = Profile(links={}, location_rules={}, seniority_rules={}, remote_required=False)
        session.add(profile)

    profile.full_name = structured.full_name or ""
    profile.email = structured.email or ""
    profile.seniority = normalize_level(structured.seniority)
    profile.years_experience = structured.years_experience
    profile.target_titles = structured.target_titles
    profile.domains_keywords = {
        "tech_stack": structured.tech_stack,
        "domains": structured.domains,
    }
    profile.work_history = [item.model_dump() for item in structured.work_history]
    profile.cv_text = text
    profile.cv_embedding = embedding

    await session.commit()
    logger.info("Profile %s", "created" if created else "updated")
    return profile
