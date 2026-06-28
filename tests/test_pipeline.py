import uuid
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.models import Job
from job_radar.ingest.base import NormalizedJob, SourceAdapter
from job_radar.ingest.normalize import content_hash
from job_radar.ingest.pipeline import run_ingestion


class _FakeAdapter(SourceAdapter):
    source = "fake"
    source_type = "board"

    def __init__(self, raw_postings: list[dict]) -> None:
        self._raw_postings = raw_postings

    async def fetch(self) -> list[dict]:
        return self._raw_postings

    def map(self, raw: dict) -> NormalizedJob:
        return NormalizedJob(**raw)


def _raw(*, url: str, title="Senior Engineer", company="Acme", location="Worldwide") -> dict:
    return {
        "source": "fake",
        "source_type": "board",
        "source_id": "1",
        "url": url,
        "title": title,
        "company": company,
        "description": "desc",
        "salary_min": None,
        "salary_max": None,
        "currency": None,
        "location": location,
        "job_type": "full_time",
        "remote": True,
        "published_at": None,
    }


def _url(name: str) -> str:
    return f"https://example.com/jobs/{name}-{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_embed(text: str) -> list[float]:
        return [0.0] * 768

    monkeypatch.setattr("job_radar.ingest.pipeline.embed", _fake_embed)


@pytest.fixture
async def _cleanup_urls(db_session: AsyncSession):
    urls: list[str] = []
    yield urls
    await db_session.execute(delete(Job).where(Job.url.in_(urls)))
    await db_session.commit()


async def test_run_ingestion_inserts_new_jobs(db_session: AsyncSession, _cleanup_urls) -> None:
    url = _url("new")
    _cleanup_urls.append(url)
    adapter = _FakeAdapter([_raw(url=url)])

    await run_ingestion(adapter, db_session, ingested_via="scheduler")

    result = await db_session.execute(select(Job).where(Job.url == url))
    job = result.scalar_one()
    assert job.title == "Senior Engineer"
    assert job.ingested_via == "scheduler"
    assert list(job.embedding) == [0.0] * 768


async def test_run_ingestion_skips_jobs_with_existing_content_hash(
    db_session: AsyncSession, _cleanup_urls
) -> None:
    existing_url = _url("existing")
    skipped_url = _url("skipped")
    _cleanup_urls.extend([existing_url, skipped_url])

    raw = _raw(url=skipped_url, title="Duplicate Engineer", company="DupeCo")
    existing = Job(
        source="fake",
        source_type="board",
        ingested_via="scheduler",
        url=existing_url,
        title=raw["title"],
        company=raw["company"],
        description="desc",
        remote=True,
        location=raw["location"],
        collected_at=datetime.now(UTC),
        embedding=[0.0] * 768,
        content_hash=content_hash(NormalizedJob(**raw)),
    )
    db_session.add(existing)
    await db_session.flush()

    adapter = _FakeAdapter([raw])
    await run_ingestion(adapter, db_session, ingested_via="scheduler")

    result = await db_session.execute(select(Job).where(Job.url == skipped_url))
    assert result.scalar_one_or_none() is None


async def test_run_ingestion_skips_url_conflicts_without_aborting_batch(
    db_session: AsyncSession, _cleanup_urls
) -> None:
    conflicting_url = _url("conflict")
    new_url = _url("ok")
    _cleanup_urls.extend([conflicting_url, new_url])

    existing = Job(
        source="fake",
        source_type="board",
        ingested_via="scheduler",
        url=conflicting_url,
        title="Original title",
        company="Original Co",
        description="desc",
        remote=True,
        collected_at=datetime.now(UTC),
        embedding=[0.0] * 768,
        content_hash="unrelated-hash",
    )
    db_session.add(existing)
    await db_session.flush()

    conflicting_raw = _raw(url=conflicting_url, title="New title", company="New Co")
    new_raw = _raw(url=new_url, title="Other Role", company="Other Co")
    adapter = _FakeAdapter([conflicting_raw, new_raw])

    await run_ingestion(adapter, db_session, ingested_via="scheduler")

    conflicting_result = await db_session.execute(select(Job).where(Job.url == conflicting_url))
    assert conflicting_result.scalar_one().title == "Original title"

    new_result = await db_session.execute(select(Job).where(Job.url == new_url))
    assert new_result.scalar_one().title == "Other Role"


async def test_run_ingestion_collapses_duplicate_hashes_within_same_batch(
    db_session: AsyncSession, _cleanup_urls
) -> None:
    first_url = _url("dup1")
    second_url = _url("dup2")
    _cleanup_urls.extend([first_url, second_url])

    same_identity = {"title": "Platform Engineer", "company": "SameCo", "location": "Worldwide"}
    raw_a = _raw(url=first_url, **same_identity)
    raw_b = _raw(url=second_url, **same_identity)
    adapter = _FakeAdapter([raw_a, raw_b])

    await run_ingestion(adapter, db_session, ingested_via="scheduler")

    result = await db_session.execute(select(Job).where(Job.url.in_([first_url, second_url])))
    jobs = result.scalars().all()
    assert len(jobs) == 1


async def test_run_ingestion_uses_provided_ingested_via(
    db_session: AsyncSession, _cleanup_urls
) -> None:
    url = _url("manual")
    _cleanup_urls.append(url)
    adapter = _FakeAdapter([_raw(url=url)])

    await run_ingestion(adapter, db_session, ingested_via="manual")

    result = await db_session.execute(select(Job).where(Job.url == url))
    assert result.scalar_one().ingested_via == "manual"


async def test_run_ingestion_skips_job_whose_embedding_call_fails(
    db_session: AsyncSession, _cleanup_urls, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing_url = _url("embed-fails")
    ok_url = _url("embed-ok")
    _cleanup_urls.extend([failing_url, ok_url])

    async def _flaky_embed(text: str) -> list[float]:
        if "Flaky Role" in text:
            raise httpx.ConnectError("ollama unreachable")
        return [0.0] * 768

    monkeypatch.setattr("job_radar.ingest.pipeline.embed", _flaky_embed)

    failing_raw = _raw(url=failing_url, title="Flaky Role")
    ok_raw = _raw(url=ok_url, title="Stable Role")
    adapter = _FakeAdapter([failing_raw, ok_raw])

    await run_ingestion(adapter, db_session, ingested_via="scheduler")

    failing_result = await db_session.execute(select(Job).where(Job.url == failing_url))
    assert failing_result.scalar_one_or_none() is None

    ok_result = await db_session.execute(select(Job).where(Job.url == ok_url))
    assert ok_result.scalar_one().title == "Stable Role"
