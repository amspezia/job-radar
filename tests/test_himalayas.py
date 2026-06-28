import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from job_radar.ingest.himalayas import HimalayasAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "himalayas_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_annual_salaried_job(raw_jobs: list[dict]) -> None:
    job = HimalayasAdapter().map(raw_jobs[0])

    assert job.source == "himalayas"
    assert job.source_type == "aggregator"
    assert job.source_id == "https://himalayas.app/companies/nimbus/jobs/staff-backend-engineer"
    assert job.url == job.source_id  # applicationLink == guid
    assert job.title == "Staff Backend Engineer"
    assert job.company == "Nimbus"
    assert job.job_type == "Full Time"
    assert job.location == "United States, Canada"
    assert job.remote is True  # remote-only board
    assert job.published_at == datetime(2026, 6, 28, 0, 38, 9, tzinfo=UTC)

    # annual salary kept
    assert job.salary_min == 150000
    assert job.salary_max == 200000
    assert job.currency == "USD"

    assert "<" not in job.description
    assert job.description == "About We need a backend engineer."


def test_map_hourly_salary_is_dropped(raw_jobs: list[dict]) -> None:
    job = HimalayasAdapter().map(raw_jobs[1])

    # hourly salary is not comparable in period-less columns -> stored unknown
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None
    assert job.location is None  # empty locationRestrictions -> None


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Returns the queued responses in order, recording how many calls happened."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls = 0

    async def get(self, url: str, params: dict) -> _FakeResponse:
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


async def test_get_page_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Don't actually sleep through the backoff during the test.
    monkeypatch.setattr("job_radar.ingest.himalayas.asyncio.sleep", _noop_sleep)

    client = _FakeClient([_FakeResponse(429), _FakeResponse(200, {"jobs": [], "totalCount": 0})])
    payload = await HimalayasAdapter._get_page(client, "engineer", 1)

    assert client.calls == 2  # retried once after the 429
    assert payload == {"jobs": [], "totalCount": 0}


async def _noop_sleep(_seconds: float) -> None:
    return None
