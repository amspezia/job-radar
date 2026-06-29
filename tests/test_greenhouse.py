import json
from datetime import datetime
from pathlib import Path

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


def test_remote_jobs_keeps_only_remote_locations() -> None:
    jobs = [
        {"location": {"name": "Remote, US"}},
        {"location": {"name": "New York"}},
        {"location": None},  # missing location object
        {"location": {"name": None}},  # location present but name null
        {},  # no location key at all
    ]
    assert GreenHouseAdapter._remote_jobs(jobs) == [{"location": {"name": "Remote, US"}}]
