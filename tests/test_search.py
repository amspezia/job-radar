import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job
from job_radar.retrieval import search as search_mod
from job_radar.retrieval.search import search


def _job(**over: object) -> Job:
    base: dict = {
        "source": "fake",
        "source_type": "board",
        "ingested_via": "manual",
        "url": f"https://example.com/jobs/{uuid.uuid4().hex}",
        "title": "Untitled",
        "company": "Acme",
        "description": "desc",
        "remote": True,
        "location": "Worldwide",
        "collected_at": datetime.now(UTC),
        # Non-zero so cosine_distance is well-defined (a zero vector is undefined).
        "embedding": [0.1] * 768,
        "content_hash": uuid.uuid4().hex,
    }
    base.update(over)
    return Job(**base)


@pytest.fixture
async def _cleanup_jobs(db_session: AsyncSession):
    jobs: list[Job] = []
    yield jobs
    if jobs:
        await db_session.execute(delete(Job).where(Job.id.in_([j.id for j in jobs])))
        await db_session.commit()


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep tests off Ollama; the returned vector matches the seeded jobs'.
    async def _fake_embed(text: str) -> list[float]:
        return [0.1] * 768

    monkeypatch.setattr(search_mod, "embed", _fake_embed)


async def test_search_returns_hydrated_jobs_for_a_match(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    # Nonce token isolates these from the real corpus on the FTS side.
    a = _job(title="Zzyqx Platform Engineer")
    b = _job(title="Zzyqx Data Engineer")
    db_session.add_all([a, b])
    await db_session.commit()
    _cleanup_jobs.extend([a, b])

    results = await search(db_session, "zzyqx", limit=20, pool=50)

    assert all(isinstance(job, Job) for job in results)  # hydrated, not (id, score)
    titles = {job.title for job in results}
    assert {"Zzyqx Platform Engineer", "Zzyqx Data Engineer"} <= titles


async def test_search_hydrates_in_fused_order(
    db_session: AsyncSession, _cleanup_jobs: list[Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _job(title="Zzyqx A")
    b = _job(title="Zzyqx B")
    c = _job(title="Zzyqx C")
    db_session.add_all([a, b, c])
    await db_session.commit()
    _cleanup_jobs.extend([a, b, c])

    # Both rankers agree on C, B, A — the reverse of insertion order, so a result
    # in that order can only come from honouring fused order, not DB order.
    ranking = [(c.id, 1.0), (b.id, 0.9), (a.id, 0.8)]

    async def fake_fts(
        session: object, query: object, limit: int, extra_filter: object = None
    ) -> list:
        return ranking

    async def fake_vector(
        session: object, embedding: object, limit: int, extra_filter: object = None
    ) -> list:
        return ranking

    monkeypatch.setattr(search_mod, "search_fts", fake_fts)
    monkeypatch.setattr(search_mod, "search_vector", fake_vector)

    results = await search(db_session, "anything")

    assert [job.id for job in results] == [c.id, b.id, a.id]


async def test_search_respects_limit(
    db_session: AsyncSession, _cleanup_jobs: list[Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _job(title="Zzyqx A")
    b = _job(title="Zzyqx B")
    c = _job(title="Zzyqx C")
    db_session.add_all([a, b, c])
    await db_session.commit()
    _cleanup_jobs.extend([a, b, c])

    ranking = [(a.id, 1.0), (b.id, 0.9), (c.id, 0.8)]

    async def fake_ranker(
        session: object, second: object, limit: int, extra_filter: object = None
    ) -> list:
        return ranking

    monkeypatch.setattr(search_mod, "search_fts", fake_ranker)
    monkeypatch.setattr(search_mod, "search_vector", fake_ranker)

    results = await search(db_session, "anything", limit=2)

    assert [job.id for job in results] == [a.id, b.id]


async def test_search_returns_empty_when_nothing_matches(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def empty(*args: object, **kwargs: object) -> list:
        return []

    monkeypatch.setattr(search_mod, "search_fts", empty)
    monkeypatch.setattr(search_mod, "search_vector", empty)

    assert await search(db_session, "anything") == []
