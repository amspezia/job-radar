import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job
from job_radar.retrieval.bm25 import search_bm25


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


async def test_search_bm25_ranks_title_match_above_requirements_only_match(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    # A nonce token keeps these assertions isolated from real corpus matches.
    # Title is boosted 5x vs requirements 2x, so a title hit outscores a
    # requirements-only hit for the same term.
    title_hit = _job(title="Senior Zzyqx Engineer", description="Build internal tools.")
    req_hit = _job(
        title="Office Manager",
        requirements="Coordinate the zzyqx on-call rotation.",
    )
    db_session.add_all([title_hit, req_hit])
    await db_session.commit()
    _cleanup_jobs.extend([title_hit, req_hit])

    results = await search_bm25(db_session, "zzyqx", limit=50)
    ids = [job_id for job_id, _ in results]

    assert title_hit.id in ids
    assert req_hit.id in ids
    assert ids.index(title_hit.id) < ids.index(req_hit.id)


async def test_search_bm25_does_not_index_description(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    # description is stored for fit analysis and display but excluded from
    # the BM25 index — only requirements and responsibilities are indexed.
    desc_only = _job(
        title="Office Manager",
        description="Must know zzyqx deeply and build zzyqx pipelines.",
    )
    db_session.add(desc_only)
    await db_session.commit()
    _cleanup_jobs.append(desc_only)

    ids = {job_id for job_id, _ in await search_bm25(db_session, "zzyqx", limit=50)}
    assert desc_only.id not in ids


async def test_search_bm25_matches_requirements_and_responsibilities(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    req_match = _job(title="Backend Engineer", requirements="Must know zzyqx deeply.")
    resp_match = _job(title="Backend Engineer", responsibilities="You will build zzyqx pipelines.")
    no_match = _job(title="Sales Rep", description="Close enterprise deals.")
    db_session.add_all([req_match, resp_match, no_match])
    await db_session.commit()
    _cleanup_jobs.extend([req_match, resp_match, no_match])

    ids = {job_id for job_id, _ in await search_bm25(db_session, "zzyqx", limit=50)}

    assert req_match.id in ids
    assert resp_match.id in ids
    assert no_match.id not in ids


async def test_search_bm25_excludes_non_matching_jobs(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    match = _job(title="Platform Zzyqx Engineer")
    no_match = _job(title="Sales Representative", description="Close enterprise deals.")
    db_session.add_all([match, no_match])
    await db_session.commit()
    _cleanup_jobs.extend([match, no_match])

    ids = {job_id for job_id, _ in await search_bm25(db_session, "zzyqx", limit=50)}

    assert match.id in ids
    assert no_match.id not in ids


async def test_search_bm25_respects_limit(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    jobs = [_job(title=f"Unusualtermzz Engineer {i}") for i in range(5)]
    db_session.add_all(jobs)
    await db_session.commit()
    _cleanup_jobs.extend(jobs)

    results = await search_bm25(db_session, "unusualtermzz", limit=2)

    assert len(results) == 2


async def test_search_bm25_partial_match_scores_without_all_terms(
    db_session: AsyncSession, _cleanup_jobs: list[Job]
) -> None:
    # BM25 partial-match: a posting matching 1 of 3 query terms still scores.
    # This is the key difference vs the old ts_rank implicit-AND that returned
    # zero rows when any term was absent.
    partial = _job(title="Backend Zzyqx Developer", description="Build APIs.")
    db_session.add_all([partial])
    await db_session.commit()
    _cleanup_jobs.extend([partial])

    # Multi-term query: partial has only 'zzyqx', not 'platform' or 'kafka'
    ids = {job_id for job_id, _ in await search_bm25(db_session, "zzyqx platform kafka", limit=50)}

    assert partial.id in ids
