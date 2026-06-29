import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job
from job_radar.retrieval.fts import search_fts


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
        "embedding": [0.0] * 768,
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


async def test_search_fts_ranks_title_match_above_description_only_match(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    # A nonce token keeps these assertions isolated from the thousands of real
    # "engineer" matches already in the corpus.
    title_hit = _job(title="Senior Zzyqx Engineer", description="Build internal tools.")
    desc_hit = _job(title="Office Manager", description="Coordinate the zzyqx on-call rotation.")
    db_session.add_all([title_hit, desc_hit])
    await db_session.commit()
    _cleanup_jobs.extend([title_hit, desc_hit])

    results = await search_fts(db_session, "zzyqx", limit=50)
    ids = [job_id for job_id, _ in results]

    assert title_hit.id in ids
    assert desc_hit.id in ids
    assert ids.index(title_hit.id) < ids.index(desc_hit.id)  # title (A) outranks description (B)


async def test_search_fts_excludes_non_matching_jobs(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    match = _job(title="Platform Zzyqx Engineer")
    no_match = _job(title="Sales Representative", description="Close enterprise deals.")
    db_session.add_all([match, no_match])
    await db_session.commit()
    _cleanup_jobs.extend([match, no_match])

    ids = {job_id for job_id, _ in await search_fts(db_session, "zzyqx", limit=50)}

    assert match.id in ids
    assert no_match.id not in ids


async def test_search_fts_respects_limit(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    jobs = [_job(title=f"Unusualtermzz Engineer {i}") for i in range(5)]
    db_session.add_all(jobs)
    await db_session.commit()
    _cleanup_jobs.extend(jobs)

    results = await search_fts(db_session, "unusualtermzz", limit=2)

    assert len(results) == 2
