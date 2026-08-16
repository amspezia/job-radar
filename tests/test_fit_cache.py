"""Tests for fit/cache.py — the judgment cache and its invalidation key.

Uses the real DB (same pattern as test_eval_qrels.py). The cache exists to skip
LLM calls, so what matters is that it misses whenever the judgment it holds could
be wrong, and hits only when it is provably still valid.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import FitJudgmentCache, Job, Profile
from job_radar.fit import cache, pipeline
from job_radar.fit.analyze import PROMPT_VERSION
from job_radar.fit.schema import DomainJudgment, Evidence, FitJudgment, Requirement
from job_radar.fit.score import score_fit

_MODEL = "test-model"


def _job(**over: object) -> Job:
    base: dict = {
        "source": "fake",
        "source_type": "board",
        "ingested_via": "manual",
        "url": f"https://example.com/jobs/{uuid.uuid4().hex}",
        "title": "Backend Engineer",
        "company": "Acme",
        "description": "desc",
        "remote": True,
        "location": "Worldwide",
        "collected_at": datetime.now(UTC),
        "content_hash": uuid.uuid4().hex,
    }
    base.update(over)
    return Job(**base)


def _profile(**over: object) -> Profile:
    base: dict = {
        "full_name": "Test User",
        "email": "test@example.com",
        "links": {},
        "work_history": [],
        "cv_text": "Backend engineer.",
        "target_titles": ["Backend Engineer"],
        "seniority": "senior",
        "domains_keywords": {"tech_stack": ["python"], "domains": ["saas"]},
        "location_rules": {},
        "remote_required": False,
    }
    base.update(over)
    return Profile(**base)


def _judgment(satisfaction: str = "met") -> FitJudgment:
    return FitJudgment(
        requirements=[
            Requirement(
                kind="required",
                satisfaction=satisfaction,
                evidence=[Evidence(source="posting", quote="5 years of Python")],
            )
        ],
        domain=DomainJudgment(relevance="strong", evidence=[]),
        summary="Strong match.",
    )


@pytest.fixture
async def _db_objects(db_session: AsyncSession):
    profile = _profile()
    job = _job()
    db_session.add_all([profile, job])
    await db_session.commit()

    yield profile, job

    await db_session.execute(
        delete(FitJudgmentCache).where(FitJudgmentCache.profile_id == profile.id)
    )
    await db_session.execute(delete(Job).where(Job.id == job.id))
    await db_session.execute(delete(Profile).where(Profile.id == profile.id))
    await db_session.commit()


async def test_roundtrip_returns_an_equal_judgment(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    profile, job = _db_objects
    original = _judgment()
    await cache.store(db_session, profile.id, [(job, original)], model=_MODEL)

    loaded = await cache.load(db_session, profile.id, [job], model=_MODEL)
    assert loaded[job.id] == original


async def test_empty_cache_is_a_miss(db_session: AsyncSession, _db_objects: tuple) -> None:
    profile, job = _db_objects
    assert await cache.load(db_session, profile.id, [job], model=_MODEL) == {}


async def test_edited_posting_misses(db_session: AsyncSession, _db_objects: tuple) -> None:
    # content_hash is the posting's identity: judging text that no longer exists
    # is exactly the stale read the cache must not serve.
    profile, job = _db_objects
    await cache.store(db_session, profile.id, [(job, _judgment())], model=_MODEL)

    job.content_hash = uuid.uuid4().hex
    await db_session.commit()

    assert await cache.load(db_session, profile.id, [job], model=_MODEL) == {}


async def test_different_model_misses(db_session: AsyncSession, _db_objects: tuple) -> None:
    profile, job = _db_objects
    await cache.store(db_session, profile.id, [(job, _judgment())], model=_MODEL)
    assert await cache.load(db_session, profile.id, [job], model="other-model") == {}


async def test_different_prompt_version_misses(
    db_session: AsyncSession, _db_objects: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, job = _db_objects
    await cache.store(db_session, profile.id, [(job, _judgment())], model=_MODEL)

    monkeypatch.setattr(cache, "PROMPT_VERSION", PROMPT_VERSION + 1)
    assert await cache.load(db_session, profile.id, [job], model=_MODEL) == {}


async def test_other_profile_misses(db_session: AsyncSession, _db_objects: tuple) -> None:
    profile, job = _db_objects
    await cache.store(db_session, profile.id, [(job, _judgment())], model=_MODEL)
    assert await cache.load(db_session, uuid.uuid4(), [job], model=_MODEL) == {}


async def test_store_overwrites_on_the_same_key(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    # What --refresh relies on: a re-run replaces the row rather than conflicting.
    profile, job = _db_objects
    await cache.store(db_session, profile.id, [(job, _judgment("met"))], model=_MODEL)
    await cache.store(db_session, profile.id, [(job, _judgment("unmet"))], model=_MODEL)

    loaded = await cache.load(db_session, profile.id, [job], model=_MODEL)
    assert loaded[job.id].requirements[0].satisfaction == "unmet"


async def test_unreadable_row_is_a_miss_not_a_crash(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    # A schema change that slipped past PROMPT_VERSION must cost a re-analysis.
    profile, job = _db_objects
    db_session.add(
        FitJudgmentCache(
            profile_id=profile.id,
            job_id=job.id,
            content_hash=job.content_hash,
            model=_MODEL,
            prompt_version=PROMPT_VERSION,
            judgment={"nonsense": True},
            created_at=datetime.now(UTC),
        )
    )
    await db_session.commit()

    assert await cache.load(db_session, profile.id, [job], model=_MODEL) == {}


async def test_no_jobs_short_circuits(db_session: AsyncSession, _db_objects: tuple) -> None:
    profile, _ = _db_objects
    assert await cache.load(db_session, profile.id, [], model=_MODEL) == {}


# ---------------------------------------------------------------------------
# Pipeline integration — the point of the cache is that a hit skips the LLM.
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Stub retrieval + LLM so the test exercises only the cache path.

    Returns the call counter for analyze_fit; the caller sets the job/profile.
    """
    calls: list[uuid.UUID] = []

    async def _analyze(profile, posting, *, levels=None, model=None):
        calls.append(posting.id)
        return score_fit(_judgment(), posting, profile, levels=levels)

    monkeypatch.setattr(pipeline, "build_hyde_embedding", lambda *a, **k: _none())
    monkeypatch.setattr(pipeline, "analyze_fit", _analyze)
    return calls


async def _none():
    return None


async def test_second_run_hits_cache_and_skips_the_llm(
    db_session: AsyncSession, _db_objects: tuple, _stub_pipeline: list, monkeypatch
) -> None:
    profile, job = _db_objects
    monkeypatch.setattr(pipeline, "_load_profile", lambda _s: _profile_coro(profile))
    monkeypatch.setattr(pipeline, "search", lambda *a, **k: _jobs_coro([job]))

    first = await pipeline.run_fit_pipeline(db_session)
    assert len(_stub_pipeline) == 1

    second = await pipeline.run_fit_pipeline(db_session)
    assert len(_stub_pipeline) == 1, "cached run must not call the LLM again"
    assert second[0][1] == first[0][1], "cached assessment must equal the fresh one"


async def test_refresh_bypasses_the_cache(
    db_session: AsyncSession, _db_objects: tuple, _stub_pipeline: list, monkeypatch
) -> None:
    profile, job = _db_objects
    monkeypatch.setattr(pipeline, "_load_profile", lambda _s: _profile_coro(profile))
    monkeypatch.setattr(pipeline, "search", lambda *a, **k: _jobs_coro([job]))

    await pipeline.run_fit_pipeline(db_session)
    await pipeline.run_fit_pipeline(db_session, refresh=True)
    assert len(_stub_pipeline) == 2


async def _profile_coro(profile: Profile) -> Profile:
    return profile


async def _jobs_coro(jobs: list[Job]) -> list[Job]:
    return jobs
