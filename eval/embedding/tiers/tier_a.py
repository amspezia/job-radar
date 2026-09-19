"""Tier A: the real profile's filtered pool and its frozen HyDE texts.

One of the three modules allowed to import `job_radar.*`, and only these three of its
modules: `db.models`, `retrieval.filters` and `ingest.embed_text`. Everything reads prod through
`prod_session()` (read-only) into plain dataclasses first; `assemble()` is then pure, so the
logic is testable without a production database. No labels are read: the prod `eval_labels`
were produced by the fit LLM, not a human, so gold comes only from `label-blind`.

`build_hyde_embedding` is deliberately never called: it writes the HyDE cache back to prod.
The frozen texts are parsed straight from `profile.dense_query_cache`.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.tiers.base import (
    JobRecord,
    PersistSummary,
    TopicPayload,
    persist_topic,
)
from eval.evals_db.base import evals_session
from eval.evals_db.prod_reader import alembic_revision, prod_session, table_counts
from job_radar.db.models import Job, Profile
from job_radar.ingest.embed_text import build_embed_text
from job_radar.retrieval.filters import build_profile_filter

ORIGIN = "prod"


@dataclass(frozen=True)
class ProfileData:
    id: UUID
    full_name: str
    cv_text: str
    target_titles: list
    seniority: str
    years_experience: float | None
    domains_keywords: dict
    work_history: list
    location_rules: dict
    remote_required: bool
    salary_floor: int | None
    currency: str | None
    dense_query_cache: str | None


@dataclass(frozen=True)
class PoolJob:
    id: UUID
    source: str
    title: str
    company: str
    url: str
    location: str | None
    remote: bool | None
    seniority: str | None
    description: str
    requirements: str | None
    responsibilities: str | None
    content_hash: str | None
    embedding: list[float] | None


@dataclass
class AssemblyStats:
    pool_size: int
    hyde_count: int


# --- CV scrubbing -----------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(
    r"(?:https?://|ftp://|www\.)\S+"
    r"|\b(?:[a-z0-9-]+\.)*(?:linkedin|github|gitlab|bitbucket|twitter|medium|behance|dribbble"
    r"|stackoverflow|kaggle|codepen|leetcode|hackerrank)\.(?:com|io|org|net)\b(?:/\S*)?"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|org|dev|me|app|xyz|page)(?:\.br)?\b(?:/\S*)?",
    re.IGNORECASE,
)
# Digits with the separators a phone number is written with. No newline, so a column of
# years is never read as one number.
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t().-]{5,}\d(?!\w)")
_YEAR = re.compile(r"(?:19|20)\d{2}")
_TRAILING_PUNCTUATION = ".,;:!?)]}>\"'"


def _url_sub(match: re.Match[str]) -> str:
    text = match.group(0)
    core = text.rstrip(_TRAILING_PUNCTUATION)
    return "[url]" + text[len(core) :]


def _phone_sub(match: re.Match[str]) -> str:
    text = match.group(0)
    groups = re.findall(r"\d+", text)
    if sum(len(g) for g in groups) < 8:
        return text
    # 2019-2023, 05.2020 - 02.2024, 2023-01-15: only years and 1-2 digit parts is a date.
    if any(_YEAR.fullmatch(g) for g in groups) and all(
        len(g) <= 2 or _YEAR.fullmatch(g) for g in groups
    ):
        return text
    return "[phone]"


def scrub_cv_text(cv_text: str, full_name: str | None) -> str:
    """Remove the PII a CV carries: emails, phone-like numbers (8+ digits), URLs, name tokens.

    Placeholders are `[email]`, `[phone]`, `[url]` and `[name]`. Years, date ranges and
    percentages survive: a number is only a phone number with eight or more digits that is
    not a date. Every name token of two or more letters is removed wherever it appears, which
    is deliberately blunt: the scrubbed text feeds an LLM judge, and a leaked name is worse
    than a lost word.
    """
    text = _EMAIL.sub("[email]", cv_text)
    text = _URL.sub(_url_sub, text)
    text = _PHONE.sub(_phone_sub, text)
    tokens = {t.strip(".,") for t in re.split(r"[\s,]+", full_name or "")}
    tokens = sorted((t for t in tokens if len(t) >= 2), key=len, reverse=True)
    if not tokens:
        return text
    alt = "|".join(re.escape(t) for t in tokens)
    name = re.compile(rf"(?<!\w)(?:{alt})(?!\w)(?:[ \t]+(?:{alt})(?!\w))*", re.IGNORECASE)
    return name.sub("[name]", text)


# --- Assembly (pure) --------------------------------------------------------------------------


def _hyde_texts(cache: str | None) -> list[str]:
    hint = "run `job-radar-fit` once to populate profile.dense_query_cache"
    if not cache:
        raise ValueError(f"the profile has no frozen HyDE texts: {hint}")
    try:
        parsed = json.loads(cache)
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile.dense_query_cache is not valid JSON: {hint}") from exc
    texts = [t for t in parsed if isinstance(t, str) and t] if isinstance(parsed, list) else []
    if not texts:
        raise ValueError(f"profile.dense_query_cache holds no HyDE texts: {hint}")
    return texts


def _snapshot(profile: ProfileData) -> dict:
    keywords = profile.domains_keywords or {}
    history = profile.work_history if isinstance(profile.work_history, list) else []
    return {
        "target_titles": list(profile.target_titles or []),
        "seniority": profile.seniority,
        "years_experience": profile.years_experience,
        "tech_stack": list(keywords.get("tech_stack", [])),
        "domains": list(keywords.get("domains", [])),
        "location_rules": profile.location_rules,
        "remote_required": profile.remote_required,
        "salary_floor": profile.salary_floor,
        "currency": profile.currency,
        "work_history": [
            {"role": entry.get("role"), "years": entry.get("years")}
            for entry in history
            if isinstance(entry, dict)
        ],
        "cv_text": scrub_cv_text(profile.cv_text, profile.full_name),
    }


def _job_record(job: PoolJob) -> JobRecord:
    embed_text = build_embed_text(
        job.title, job.description, job.requirements, job.responsibilities
    )
    return JobRecord(
        origin=ORIGIN,
        origin_id=str(job.id),
        source=job.source,
        title=job.title,
        company=job.company,
        url=job.url,
        location=job.location,
        remote=job.remote,
        seniority=job.seniority,
        description=job.description,
        requirements=job.requirements,
        responsibilities=job.responsibilities,
        content_hash=job.content_hash,
        embed_text=embed_text,
        embed_text_sha=hashlib.sha256(embed_text.encode()).hexdigest(),
        prod_embedding=job.embedding,
    )


def assemble(
    name: str, profile: ProfileData, jobs: Sequence[PoolJob]
) -> tuple[TopicPayload, AssemblyStats]:
    """Turn fetched prod rows into a topic payload. Pure: no I/O.

    `jobs` is the already-filtered pool. The payload carries no labels.
    """
    hyde = _hyde_texts(profile.dense_query_cache)
    records = [_job_record(job) for job in jobs]
    stats = AssemblyStats(pool_size=len(records), hyde_count=len(hyde))
    payload = TopicPayload(
        tier="A",
        name=name,
        profile_snapshot=_snapshot(profile),
        query_inputs={"hyde_texts": hyde},
        jobs=records,
        labels=[],
        builder="tier_a",
        notes=f"prod profile {profile.id}",
    )
    return payload, stats


# --- Fetching (read-only) ---------------------------------------------------------------------


async def _load_profile(session: AsyncSession, profile_id: UUID | None) -> Profile:
    if profile_id is not None:
        profile = await session.get(Profile, profile_id)
        if profile is None:
            raise LookupError(f"no profile with id {profile_id}")
        return profile
    real = (await session.scalars(select(Profile).where(Profile.source == "real"))).all()
    if len(real) != 1:
        raise LookupError(
            f"expected exactly one real profile, found {len(real)}; pass profile_id to choose"
        )
    return real[0]


def _profile_data(profile: Profile) -> ProfileData:
    return ProfileData(
        id=profile.id,
        full_name=profile.full_name,
        cv_text=profile.cv_text,
        target_titles=list(profile.target_titles or []),
        seniority=profile.seniority,
        years_experience=profile.years_experience,
        domains_keywords=dict(profile.domains_keywords or {}),
        work_history=list(profile.work_history or []),
        location_rules=dict(profile.location_rules or {}),
        remote_required=profile.remote_required,
        salary_floor=profile.salary_floor,
        currency=profile.currency,
        dense_query_cache=profile.dense_query_cache,
    )


def _pool_job(job: Job) -> PoolJob:
    return PoolJob(
        id=job.id,
        source=job.source,
        title=job.title,
        company=job.company,
        url=job.url,
        location=job.location,
        remote=job.remote,
        seniority=job.seniority,
        description=job.description,
        requirements=job.requirements,
        responsibilities=job.responsibilities,
        content_hash=job.content_hash,
        embedding=None if job.embedding is None else [float(x) for x in job.embedding],
    )


class TierABuilder:
    tier = "A"

    def __init__(self, prod_url: str | None = None) -> None:
        self._prod_url = prod_url

    async def build(self, name: str, profile_id: UUID | str | None = None) -> TopicPayload:
        async with prod_session(self._prod_url) as session:
            payload, _ = await self.fetch_and_assemble(session, name, profile_id)
        return payload

    async def fetch_and_assemble(
        self, session: AsyncSession, name: str, profile_id: UUID | str | None = None
    ) -> tuple[TopicPayload, AssemblyStats]:
        """Read prod through `session` (which must be read-only) and assemble the topic."""
        profile = await _load_profile(
            session, None if profile_id is None else UUID(str(profile_id))
        )
        query = select(Job).where(Job.embedding.is_not(None)).order_by(Job.id)
        profile_filter = build_profile_filter(profile)
        if profile_filter is not None:
            query = query.where(profile_filter)
        pool = [_pool_job(job) for job in (await session.scalars(query)).all()]
        return assemble(name, _profile_data(profile), pool)


async def _prod_state(session: AsyncSession) -> tuple[dict[str, int], str | None]:
    await session.rollback()  # a fresh snapshot, not the transaction the reads ran in
    return await table_counts(session), await alembic_revision(session)


async def import_topic(
    name: str,
    tier: str,
    *,
    profile_id: UUID | str | None = None,
    prod_url: str | None = None,
    evals_url: str | None = None,
) -> PersistSummary:
    """Build the Tier-A topic from prod and persist it in the EVALS database.

    Prod row counts and Alembic revision are read before and after; a difference aborts before
    anything is stored, because the harness must never have changed production.
    """
    if tier.upper() != "A":
        raise ValueError(f"tier {tier!r} is not supported by this builder; expected 'A'")
    async with prod_session(prod_url) as prod:
        before = await _prod_state(prod)
        payload, stats = await TierABuilder(prod_url).fetch_and_assemble(prod, name, profile_id)
        after = await _prod_state(prod)
    if before != after:
        raise RuntimeError(f"production changed during the import: {before} -> {after}")

    async with evals_session(evals_url) as session:
        summary = await persist_topic(session, payload)

    print(
        f"topic {name!r} frozen: {summary.n_jobs_new} new documents, {summary.n_jobs_reused} reused"
    )
    print(f"  pool size:        {stats.pool_size}")
    print(f"  HyDE texts:       {stats.hyde_count}")
    print(f"  prod unchanged:   {before[0]} at revision {before[1]}")
    return summary
