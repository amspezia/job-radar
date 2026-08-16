"""Tests for fit/pipeline.py's HyDE posting cache (Profile.dense_query_cache).

Uses the real DB (same pattern as test_fit_cache.py) since the point of this cache
is a real commit that a later call must actually see.
"""

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Profile
from job_radar.fit import pipeline


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


@pytest.fixture
async def _db_profile(db_session: AsyncSession):
    profile = _profile()
    db_session.add(profile)
    await db_session.commit()

    yield profile

    await db_session.execute(delete(Profile).where(Profile.id == profile.id))
    await db_session.commit()


@pytest.fixture
def _stub_generation(monkeypatch: pytest.MonkeyPatch):
    """Stub HyDE synthesis + embedding; returns the call counters."""
    gen_calls: list[int] = []
    embed_calls: list[str] = []

    async def _fake_generate(profile: Profile) -> str:
        gen_calls.append(1)
        return f"synthetic posting {len(gen_calls)}"

    async def _fake_embed(text: str, *, task: str) -> list[float]:
        embed_calls.append(text)
        return [float(len(text))] * 3

    monkeypatch.setattr(pipeline, "_generate_hyde_posting", _fake_generate)
    monkeypatch.setattr(pipeline, "embed", _fake_embed)
    return gen_calls, embed_calls


async def test_cache_miss_generates_and_persists(
    db_session: AsyncSession, _db_profile: Profile, _stub_generation: tuple
) -> None:
    gen_calls, _ = _stub_generation
    assert _db_profile.dense_query_cache is None

    result = await pipeline.build_hyde_embedding(_db_profile, db_session)

    assert result is not None
    assert len(gen_calls) == pipeline._HYDE_N
    assert _db_profile.dense_query_cache is not None


async def test_cache_hit_skips_generation(
    db_session: AsyncSession, _db_profile: Profile, _stub_generation: tuple
) -> None:
    gen_calls, embed_calls = _stub_generation
    await pipeline.build_hyde_embedding(_db_profile, db_session)
    assert len(gen_calls) == pipeline._HYDE_N
    embed_calls.clear()

    result = await pipeline.build_hyde_embedding(_db_profile, db_session)

    assert result is not None
    assert len(gen_calls) == pipeline._HYDE_N, "second call must not synthesize again"
    assert len(embed_calls) == pipeline._HYDE_N, "cache hit still embeds all N cached texts"


async def test_cache_reload_returns_the_same_texts(
    db_session: AsyncSession, _db_profile: Profile, _stub_generation: tuple
) -> None:
    """A cache hit must not silently collapse to one sample (losing variance reduction)."""
    await pipeline.build_hyde_embedding(_db_profile, db_session)
    cached_after_first = pipeline._load_cached_hyde_texts(_db_profile)
    assert len(cached_after_first) == pipeline._HYDE_N

    await pipeline.build_hyde_embedding(_db_profile, db_session)
    cached_after_second = pipeline._load_cached_hyde_texts(_db_profile)
    assert cached_after_second == cached_after_first


async def test_cv_reload_invalidation_forces_regeneration(
    db_session: AsyncSession, _db_profile: Profile, _stub_generation: tuple
) -> None:
    gen_calls, _ = _stub_generation
    await pipeline.build_hyde_embedding(_db_profile, db_session)
    assert len(gen_calls) == pipeline._HYDE_N

    _db_profile.dense_query_cache = None  # what profile/loader.py does on CV reload
    await db_session.commit()

    await pipeline.build_hyde_embedding(_db_profile, db_session)
    assert len(gen_calls) == 2 * pipeline._HYDE_N


async def test_unreadable_cache_falls_back_to_regenerating(
    db_session: AsyncSession, _db_profile: Profile, _stub_generation: tuple
) -> None:
    gen_calls, _ = _stub_generation
    _db_profile.dense_query_cache = "not valid json"
    await db_session.commit()

    result = await pipeline.build_hyde_embedding(_db_profile, db_session)

    assert result is not None
    assert len(gen_calls) == pipeline._HYDE_N


async def test_all_generation_failures_return_none_without_caching(
    db_session: AsyncSession, _db_profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _failing_generate(profile: Profile) -> str | None:
        return None

    monkeypatch.setattr(pipeline, "_generate_hyde_posting", _failing_generate)

    result = await pipeline.build_hyde_embedding(_db_profile, db_session)

    assert result is None
    assert _db_profile.dense_query_cache is None


def test_load_cached_hyde_texts_ignores_non_string_entries() -> None:
    profile = _profile()
    profile.dense_query_cache = '["a valid posting", 42, null, ""]'

    assert pipeline._load_cached_hyde_texts(profile) == ["a valid posting"]


def test_load_cached_hyde_texts_empty_when_unset() -> None:
    profile = _profile()
    assert pipeline._load_cached_hyde_texts(profile) == []
