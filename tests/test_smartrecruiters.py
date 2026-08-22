import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from job_radar.adapters.sources.smartrecruiters import SmartRecruitersAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "smartrecruiters_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    """Postings in the merged list+detail shape `fetch()` hands to `map()`."""
    return json.loads(_FIXTURE.read_text())


def test_map_remote_job(raw_jobs: list[dict]) -> None:
    job = SmartRecruitersAdapter().map(raw_jobs[0])

    assert job.source == "smartrecruiters"
    assert job.source_type == "board"
    assert job.source_id == "744000144839749"
    assert job.url.endswith("744000144839749-senior-backend-engineer")
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme Labs"
    assert job.location == "Lisbon, Lisboa, Portugal"
    assert job.job_type == "Full-time"
    assert job.remote is True
    assert job.published_at == datetime.fromisoformat("2026-04-20T11:24:43.000+00:00")

    # No salary field is exposed on the public API.
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None


def test_description_joins_only_requirement_bearing_sections(raw_jobs: list[dict]) -> None:
    description = SmartRecruitersAdapter().map(raw_jobs[0]).description

    assert description == "Build backend services. 5+ years Python."
    assert "<" not in description
    assert "We are Acme Labs" not in description  # companyDescription dropped
    assert "Great benefits" not in description  # additionalInformation dropped


def test_description_tolerates_an_empty_section(raw_jobs: list[dict]) -> None:
    # qualifications is "" here, and an unexpected "videos" section is present.
    assert SmartRecruitersAdapter().map(raw_jobs[2]).description == "Own the pipeline ."


def test_description_is_empty_when_no_sections_exist() -> None:
    assert SmartRecruitersAdapter._description({"jobAd": {}}) == ""


def test_map_handles_missing_released_date(raw_jobs: list[dict]) -> None:
    assert SmartRecruitersAdapter().map(raw_jobs[2]).published_at is None


async def test_list_remote_pages_and_keeps_only_remote(
    monkeypatch: pytest.MonkeyPatch, raw_jobs: list[dict]
) -> None:
    pages = [raw_jobs, []]  # second page empty -> paging stops
    requested: list[dict] = []

    class _Resp:
        def __init__(self, body: dict) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._body

    class _Client:
        async def get(self, url: str, params: dict | None = None) -> _Resp:
            requested.append(params or {})
            return _Resp({"content": pages.pop(0)})

    remote = await SmartRecruitersAdapter._list_remote(_Client(), "acmelabs")

    assert [p["name"] for p in remote] == ["Senior Backend Engineer", "Data Engineer"]
    # offset advances by the number of rows actually returned
    assert [p["offset"] for p in requested] == [0, 3]


async def test_detail_merges_detail_over_the_list_item() -> None:
    import asyncio

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"id": "1", "postingUrl": "https://example.com/1", "jobAd": {"sections": {}}}

    class _Client:
        async def get(self, url: str) -> _Resp:
            return _Resp()

    merged = await SmartRecruitersAdapter._detail(
        _Client(), asyncio.Semaphore(1), {"id": "1", "ref": "https://api/1", "name": "Role"}
    )

    assert merged["name"] == "Role"  # kept from the list item
    assert merged["postingUrl"] == "https://example.com/1"  # added by the detail call


async def test_detail_returns_none_on_http_error_without_raising() -> None:
    import asyncio

    class _Client:
        async def get(self, url: str) -> object:
            raise httpx.ConnectError("boom", request=httpx.Request("GET", url))

    result = await SmartRecruitersAdapter._detail(
        _Client(), asyncio.Semaphore(1), {"id": "1", "ref": "https://api/1"}
    )

    assert result is None
