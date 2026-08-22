import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from job_radar.adapters.sources.greenhouse import GreenHouseAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "greenhouse_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_job_with_pay_range_salary(raw_jobs: list[dict]) -> None:
    job = GreenHouseAdapter().map(raw_jobs[0])

    assert job.source == "greenhouse"
    assert job.source_type == "board"
    assert job.source_id == "8503792002"
    assert job.url == "https://job-boards.greenhouse.io/acme/jobs/8503792002"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme"
    assert job.location == "Remote, US"
    assert job.remote is True
    assert job.published_at == datetime.fromisoformat("2026-04-17T05:58:03-04:00")

    # salary parsed from the escaped pay-range element
    assert job.salary_min == 108400
    assert job.salary_max == 129600
    assert job.currency == "USD"

    # content was HTML-escaped: unescaped, then tags stripped
    assert "<" not in job.description
    assert "&lt;" not in job.description
    assert "Build backend services." in job.description


def test_map_job_without_pay_range_has_no_salary(raw_jobs: list[dict]) -> None:
    job = GreenHouseAdapter().map(raw_jobs[1])

    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None
    # get_text(separator=" ") inserts a space at the <em> boundary (harmless).
    assert job.description == "Own the roadmap ."


def test_map_job_folds_target_country_metadata_into_location(raw_jobs: list[dict]) -> None:
    job = GreenHouseAdapter().map(raw_jobs[2])

    assert job.location == "Remote - Romania, EMEA, Brazil, Poland, Romania"


def test_map_job_without_metadata_keeps_bare_location(raw_jobs: list[dict]) -> None:
    job = GreenHouseAdapter().map(raw_jobs[0])

    assert job.location == "Remote, US"


def test_remote_jobs_keeps_only_remote_locations() -> None:
    jobs = [
        {"location": {"name": "Remote, US"}},
        {"location": {"name": "New York"}},
        {"location": None},  # missing location object
        {"location": {"name": None}},  # location present but name null
        {},  # no location key at all
    ]
    assert GreenHouseAdapter._remote_jobs(jobs) == [{"location": {"name": "Remote, US"}}]


async def test_board_returns_only_remote_jobs() -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "jobs": [
                    {"location": {"name": "Remote"}},
                    {"location": {"name": "New York"}},
                ]
            }

    class _Client:
        async def get(self, url: str, params: dict | None = None) -> _Resp:
            return _Resp()

    jobs = await GreenHouseAdapter._board(_Client(), asyncio.Semaphore(1), "acme")

    assert jobs == [{"location": {"name": "Remote"}}]


async def test_board_returns_empty_list_on_http_error_without_raising() -> None:
    class _Client:
        async def get(self, url: str, params: dict | None = None) -> object:
            raise httpx.ConnectError("boom", request=httpx.Request("GET", url))

    jobs = await GreenHouseAdapter._board(_Client(), asyncio.Semaphore(1), "acme")

    assert jobs == []
