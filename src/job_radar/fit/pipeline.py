import asyncio
import logging

from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.adapters.embeddings import embed
from job_radar.adapters.generation import generate
from job_radar.db.models import Job, Profile
from job_radar.fit.analyze import analyze_fit
from job_radar.fit.schema import FitAssessment
from job_radar.retrieval.filters import build_profile_filter
from job_radar.retrieval.search import search

logger = logging.getLogger(__name__)

# Caps how many analyze_fit calls run at once. Matches Ollama's typical default
# OLLAMA_NUM_PARALLEL; unbounded concurrency would just queue identically to
# sequential (or exhaust GPU VRAM) instead of actually overlapping.
_MAX_CONCURRENT_ANALYSES = 2


class _HyDEPosting(BaseModel):
    """Single-field schema that forces the model to return a plain prose paragraph.

    Structured output (JSON schema mode) eliminates bullet points, headers, and
    markdown that would degrade the embedding. The length check catches the model
    producing a one-liner instead of a rich posting body.
    """

    description: str

    @field_validator("description")
    @classmethod
    def _check_length(cls, v: str) -> str:
        words = v.split()
        if len(words) < 60:
            raise ValueError(f"posting too short ({len(words)} words, minimum 60)")
        return v


# HyDE: generate a synthetic job posting from the employer's perspective so the
# resulting embedding lives in the same semantic space as indexed job descriptions.
# The model writes "You will…" / "We are looking for…" — not "the candidate wants…"
# — so the vector aligns with document embeddings, not query embeddings.
_HYDE_PROMPT = """\
Write 8-10 sentences of a job posting body for the role this candidate is best suited for.
Write as the hiring company, in second person ("You will...", "The ideal candidate...",
"We are looking for..."). Include: the seniority level, 3-5 specific technologies by name,
two or three concrete day-to-day responsibilities, and the business domain.
No bullet points. No markdown. No company name. Return only the posting body paragraph.

Candidate profile:
seniority: {seniority}
target roles: {target_titles}
tech stack: {tech_stack}
domains: {domains}
"""


async def _load_profile(session: AsyncSession) -> Profile | None:
    return (await session.execute(select(Profile))).scalars().first()


def build_lexical_query(profile: Profile) -> str:
    """Keyword bag for the BM25 arm.

    Titles + stack + domains fed to BM25 partial-match scoring; each token
    contributes independently so adding more terms broadens rather than narrows.
    """
    keywords = profile.domains_keywords or {}
    parts = [
        *(profile.target_titles or []),
        *keywords.get("tech_stack", []),
        *keywords.get("domains", []),
    ]
    return " ".join(parts)


async def build_dense_query(profile: Profile, session: AsyncSession) -> str:
    """Synthesize a HyDE job posting for the semantic retrieval arm, cached per profile.

    Generates a synthetic job description from the employer's perspective and
    stores it in `profile.dense_query_cache`. Subsequent calls return the cached
    text. Cache is cleared by `load_profile` when the CV is reloaded.

    Returns empty string on generation failure; the caller skips the HyDE arm
    in that case rather than aborting the whole pipeline.
    """
    if profile.dense_query_cache:
        return profile.dense_query_cache

    keywords = profile.domains_keywords or {}
    prompt = _HYDE_PROMPT.format(
        seniority=profile.seniority or "unspecified",
        target_titles=", ".join(profile.target_titles or []),
        tech_stack=", ".join(keywords.get("tech_stack", [])),
        domains=", ".join(keywords.get("domains", [])),
    )
    logger.info("Synthesizing HyDE posting via LLM (first run or cache invalidated)")
    try:
        posting = await generate(prompt, _HyDEPosting)
        text = posting.description
    except Exception:
        logger.warning("HyDE posting synthesis failed; dense arm will be skipped this run")
        return ""

    profile.dense_query_cache = text
    await session.commit()
    logger.info("HyDE posting cached (%d words)", len(text.split()))
    return text


async def run_fit_pipeline(
    session: AsyncSession,
    query: str | None = None,
    *,
    limit: int = 20,
    levels: list[str] | None = None,
) -> list[tuple[Job, FitAssessment]]:
    """Retrieve candidate jobs for the stored profile and score each one's fit.

    Results are sorted best-first; jobs the pre-flight guard skipped (score is
    None) sort last. `levels` overrides the profile's accepted seniority levels
    for this run, filtering retrieval and gating scoring alike.
    """
    profile = await _load_profile(session)
    if profile is None:
        raise ValueError("no profile loaded — run job-radar-profile first")

    lexical_q = query or build_lexical_query(profile)
    dense_text = await build_dense_query(profile, session)
    hyde_embedding = await embed(dense_text, task="document") if dense_text else None
    profile_filter = build_profile_filter(profile, levels=levels)
    logger.info(
        "Searching: lexical=%r hyde=%s filtered=%s",
        lexical_q,
        f"{len(dense_text.split())}w" if dense_text else "skipped",
        profile_filter is not None,
    )
    jobs = await search(
        session,
        lexical_q,
        hyde_embedding=hyde_embedding,
        limit=limit,
        extra_filter=profile_filter,
        profile_embedding=profile.cv_embedding,
    )
    logger.info("Retrieved %d candidate jobs", len(jobs))

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)

    async def _bounded_analyze(job: Job) -> tuple[Job, FitAssessment]:
        async with semaphore:
            return job, await analyze_fit(profile, job, levels=levels)

    results = await asyncio.gather(*(_bounded_analyze(job) for job in jobs))
    return sorted(
        results, key=lambda pair: pair[1].score if pair[1].score is not None else -1, reverse=True
    )
