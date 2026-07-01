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
_MAX_CONCURRENT_ANALYSES = 12


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
Write 8-10 sentences of a job posting body for a {seniority}-level role in one of these areas:
{target_titles}

Write as the hiring company, in second person ("You will...", "The ideal candidate...",
"We are looking for..."). Use the vocabulary and framing that appears in real job postings
for these roles — not the vocabulary of the candidate's past work.

Rules:
- Anchor the posting to the TARGET ROLES above, not to the candidate's past projects.
- Use the work history ONLY to infer the candidate's depth and strongest technologies.
  Do NOT copy project descriptions, internal tool names, or company-specific patterns.
- Name 4-6 technologies that a strong candidate for these roles would know.
- Describe 3-4 day-to-day responsibilities typical of these roles at this seniority.
- Prefer general, employer-facing language ("design and ship backend services",
  "build LLM-powered features", "own the data pipeline") over implementation details.
- No bullet points. No markdown. No company name. Return only the posting body paragraph.

Candidate's demonstrated stack and experience (use to infer skill depth only):
tech stack: {tech_stack}

Recent work history (most recent first — use for skill-level calibration, not as source material):
{work_history}
"""


async def _load_profile(session: AsyncSession) -> Profile | None:
    return (await session.execute(select(Profile))).scalars().first()


def build_lexical_query(profile: Profile) -> str:
    """Keyword bag for the BM25 arm.

    Titles + stack fed to BM25 partial-match scoring. Tokens are deduplicated
    to prevent repeated words (e.g. "Engineer" from multiple target titles)
    from accumulating artificial IDF weight and skewing BM25 scores.
    """
    keywords = profile.domains_keywords or {}
    parts = [
        *(profile.target_titles or []),
        *keywords.get("tech_stack", []),
    ]
    seen: set[str] = set()
    tokens: list[str] = []
    for part in parts:
        for tok in part.split():
            if tok.lower() not in seen:
                seen.add(tok.lower())
                tokens.append(tok)
    return " ".join(tokens)


# Postings averaged per query; multiple independent samples smooth LLM
# token-sampling variance so the mean embedding sits closer to the centroid
# of the target-posting cluster than any single draw would.
_HYDE_N = 3


async def _generate_hyde_posting(profile: Profile) -> str | None:
    """Generate one synthetic HyDE posting. Returns None on LLM failure."""
    keywords = profile.domains_keywords or {}
    history = profile.work_history or []
    history_lines: list[str] = []
    for entry in history:
        role = entry.get("role", "")
        years = entry.get("years", "")
        line = f"- {role} ({years} yrs)" if years else f"- {role}"
        history_lines.append(line)

    prompt = _HYDE_PROMPT.format(
        seniority=profile.seniority or "unspecified",
        target_titles=", ".join(profile.target_titles or []),
        tech_stack=", ".join(keywords.get("tech_stack", [])),
        work_history="\n".join(history_lines) if history_lines else "(none provided)",
    )
    try:
        posting = await generate(prompt, _HyDEPosting)
        return posting.description
    except Exception:
        return None


async def build_hyde_embedding(
    profile: Profile, _session: AsyncSession
) -> list[float] | None:
    """Synthesize _HYDE_N HyDE postings concurrently and return their averaged embedding.

    Averaging embeddings across multiple independent LLM samples reduces the
    single-call variance that otherwise causes run-to-run ranking instability.
    Falls back to fewer samples when some calls fail; returns None only when
    every attempt fails, in which case the caller skips the dense arm entirely.
    """
    logger.info("Synthesizing %d HyDE postings concurrently", _HYDE_N)
    texts = await asyncio.gather(*(_generate_hyde_posting(profile) for _ in range(_HYDE_N)))
    valid = [t for t in texts if t]
    if not valid:
        logger.warning("All HyDE postings failed; dense arm will be skipped")
        return None
    if len(valid) < _HYDE_N:
        logger.warning(
            "%d/%d HyDE postings failed; averaging remaining %d",
            _HYDE_N - len(valid),
            _HYDE_N,
            len(valid),
        )
    embeddings = await asyncio.gather(*(embed(t, task="document") for t in valid))
    dim = len(embeddings[0])
    n = len(embeddings)
    return [sum(e[i] for e in embeddings) / n for i in range(dim)]


async def run_fit_pipeline(
    session: AsyncSession,
    query: str | None = None,
    *,
    limit: int = 50,
    levels: list[str] | None = None,
    field_boosts: dict[str, int] | None = None,
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
    hyde_embedding = await build_hyde_embedding(profile, session)
    profile_filter = build_profile_filter(profile, levels=levels)
    logger.info(
        "Searching: lexical=%r hyde=%s filtered=%s",
        lexical_q,
        "ok" if hyde_embedding else "skipped",
        profile_filter is not None,
    )
    jobs = await search(
        session,
        lexical_q,
        hyde_embedding=hyde_embedding,
        limit=limit,
        extra_filter=profile_filter,
        field_boosts=field_boosts,
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
