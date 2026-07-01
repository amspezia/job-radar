"""Tests for eval/qrels.py — the DB ↔ TREC primitive adapters.

load_qrels tests use the real DB (same pattern as test_models.py).
build_run tests mock out the arm functions to verify config wiring without
needing ParadeDB or vector search live in every CI environment.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from eval import qrels as qrels_mod
from eval.qrels import SearchConfig, build_run, load_qrels
from job_radar.db.models import EvalLabel, Job, Profile


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
        "embedding": [0.1] * 768,
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
        "cv_embedding": [0.1] * 768,
        "target_titles": ["Backend Engineer"],
        "seniority": "senior",
        "domains_keywords": {"tech_stack": ["python"], "domains": ["saas"]},
        "location_rules": {},
        "remote_required": False,
    }
    base.update(over)
    return Profile(**base)


@pytest.fixture
async def _db_objects(db_session: AsyncSession):
    """Insert a profile, two jobs, and one label; clean up after the test."""
    profile = _profile()
    job_a = _job()
    job_b = _job()
    db_session.add_all([profile, job_a, job_b])
    await db_session.flush()

    label = EvalLabel(
        profile_id=profile.id,
        job_id=job_a.id,
        label="3",
        labeled_by="test",
    )
    db_session.add(label)
    await db_session.commit()

    yield profile, job_a, job_b

    await db_session.execute(
        delete(EvalLabel).where(EvalLabel.profile_id == profile.id)
    )
    await db_session.execute(delete(Job).where(Job.id.in_([job_a.id, job_b.id])))
    await db_session.execute(delete(Profile).where(Profile.id == profile.id))
    await db_session.commit()


# ---------------------------------------------------------------------------
# load_qrels
# ---------------------------------------------------------------------------


async def test_load_qrels_returns_labeled_grade(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    profile, job_a, _ = _db_objects
    qrels = await load_qrels(db_session, profile.id)
    assert qrels[job_a.id] == 3


async def test_load_qrels_excludes_unlabeled_jobs(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    profile, _, job_b = _db_objects
    qrels = await load_qrels(db_session, profile.id)
    assert job_b.id not in qrels


async def test_load_qrels_maps_all_digit_grades(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    profile, _, job_b = _db_objects
    for grade_str, expected in [("0", 0), ("1", 1), ("2", 2), ("3", 3)]:
        label = EvalLabel(
            profile_id=profile.id,
            job_id=job_b.id,
            label=grade_str,
            labeled_by="test",
        )
        db_session.add(label)
        await db_session.commit()
        qrels = await load_qrels(db_session, profile.id)
        assert qrels[job_b.id] == expected
        await db_session.execute(
            delete(EvalLabel).where(
                EvalLabel.profile_id == profile.id, EvalLabel.job_id == job_b.id
            )
        )
        await db_session.commit()


async def test_load_qrels_also_accepts_verdict_aliases(
    db_session: AsyncSession, _db_objects: tuple
) -> None:
    profile, _, job_b = _db_objects
    for verdict, expected_grade in [
        ("strong", 3),
        ("moderate", 2),
        ("relevant", 2),
        ("weak", 1),
        ("marginal", 1),
        ("none", 0),
    ]:
        label = EvalLabel(
            profile_id=profile.id,
            job_id=job_b.id,
            label=verdict,
            labeled_by="test",
        )
        db_session.add(label)
        await db_session.commit()
        qrels = await load_qrels(db_session, profile.id)
        assert qrels[job_b.id] == expected_grade
        await db_session.execute(
            delete(EvalLabel).where(
                EvalLabel.profile_id == profile.id, EvalLabel.job_id == job_b.id
            )
        )
        await db_session.commit()


async def test_load_qrels_empty_when_no_labels(db_session: AsyncSession) -> None:
    profile = _profile()
    db_session.add(profile)
    await db_session.commit()
    qrels = await load_qrels(db_session, profile.id)
    assert qrels == {}
    await db_session.execute(delete(Profile).where(Profile.id == profile.id))
    await db_session.commit()


# ---------------------------------------------------------------------------
# build_run  (arms mocked so BM25 / vector search aren't needed here)
# ---------------------------------------------------------------------------


async def test_build_run_returns_scores_for_all_active_arms(
    db_session: AsyncSession, _db_objects: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, job_a, job_b = _db_objects
    # Fake BM25 and vector return the two jobs.
    async def fake_bm25(session, query, limit, extra_filter=None, *, field_boosts=None):
        return [(job_a.id, 9.0), (job_b.id, 8.0)]

    async def fake_vector(session, embedding, limit, extra_filter=None):
        return [(job_b.id, 0.9), (job_a.id, 0.8)]

    monkeypatch.setattr(qrels_mod, "search_bm25", fake_bm25)
    monkeypatch.setattr(qrels_mod, "search_vector", fake_vector)

    config = SearchConfig(arms=["lexical", "hyde", "cv"], pool=10, limit=10)
    run = await build_run(
        db_session,
        profile,
        config,
        query="python engineer",
        hyde_embedding=[0.1] * 768,
    )

    assert job_a.id in run
    assert job_b.id in run
    # Scores are positive RRF values.
    assert all(s > 0 for s in run.values())


async def test_build_run_respects_pool_size(
    db_session: AsyncSession, _db_objects: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, job_a, job_b = _db_objects
    captured_limits: list[int] = []

    async def fake_bm25(session, query, limit, extra_filter=None, *, field_boosts=None):
        captured_limits.append(limit)
        return [(job_a.id, 1.0)]

    async def fake_vector(session, embedding, limit, extra_filter=None):
        captured_limits.append(limit)
        return [(job_b.id, 0.9)]

    monkeypatch.setattr(qrels_mod, "search_bm25", fake_bm25)
    monkeypatch.setattr(qrels_mod, "search_vector", fake_vector)

    config = SearchConfig(arms=["lexical", "hyde", "cv"], pool=7, limit=7)
    await build_run(
        db_session, profile, config, query="test", hyde_embedding=[0.1] * 768
    )

    assert all(lim == 7 for lim in captured_limits)


async def test_build_run_skips_lexical_arm_when_query_blank(
    db_session: AsyncSession, _db_objects: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, job_a, _ = _db_objects
    bm25_called = False

    async def should_not_be_called(*args, **kwargs):
        nonlocal bm25_called
        bm25_called = True
        return []

    async def fake_vector(session, embedding, limit, extra_filter=None):
        return [(job_a.id, 0.9)]

    monkeypatch.setattr(qrels_mod, "search_bm25", should_not_be_called)
    monkeypatch.setattr(qrels_mod, "search_vector", fake_vector)

    config = SearchConfig(arms=["lexical", "hyde", "cv"], pool=10, limit=10)
    run = await build_run(
        db_session, profile, config, query="", hyde_embedding=[0.1] * 768
    )

    assert not bm25_called
    assert job_a.id in run


async def test_build_run_returns_empty_with_no_active_arms(
    db_session: AsyncSession, _db_objects: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, _, _ = _db_objects
    # No query, no hyde embedding, profile has no cv_embedding.
    profile.cv_embedding = None

    async def boom(*args, **kwargs):
        raise AssertionError("no arm should be called")

    monkeypatch.setattr(qrels_mod, "search_bm25", boom)
    monkeypatch.setattr(qrels_mod, "search_vector", boom)

    config = SearchConfig(arms=["lexical", "hyde", "cv"], pool=10, limit=10)
    run = await build_run(db_session, profile, config, query="", hyde_embedding=None)

    assert run == {}


async def test_build_run_keyword_only_config_omits_vector_arms(
    db_session: AsyncSession, _db_objects: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, job_a, _ = _db_objects
    vector_called = False

    async def fake_bm25(session, query, limit, extra_filter=None, *, field_boosts=None):
        return [(job_a.id, 5.0)]

    async def track_vector(session, embedding, limit, extra_filter=None):
        nonlocal vector_called
        vector_called = True
        return []

    monkeypatch.setattr(qrels_mod, "search_bm25", fake_bm25)
    monkeypatch.setattr(qrels_mod, "search_vector", track_vector)

    config = SearchConfig(arms=["lexical"], pool=10, limit=10)
    run = await build_run(
        db_session, profile, config, query="python", hyde_embedding=[0.1] * 768
    )

    assert not vector_called
    assert job_a.id in run
