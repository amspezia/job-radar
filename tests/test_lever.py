import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from job_radar.adapters.sources.lever import LeverAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "lever_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_job_with_annual_salary(raw_jobs: list[dict]) -> None:
    job = LeverAdapter().map(raw_jobs[0])

    assert job.source == "lever"
    assert job.source_type == "board"
    assert job.source_id == "a1b2c3d4-0000-0000-0000-000000000001"
    assert job.url == "https://jobs.lever.co/acme-corp/a1b2c3d4-0000-0000-0000-000000000001"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme Corp"  # slug token title-cased
    assert job.location == "Remote - Americas, Remote - EU"  # allLocations joined
    assert job.job_type == "Full-Time"
    assert job.remote is True
    assert job.published_at == datetime.fromtimestamp(1745000000, tz=UTC)  # epoch ms -> s

    assert job.salary_min == 120000
    assert job.salary_max == 160000
    assert job.currency == "USD"

    assert "<" not in job.description
    assert job.description == "Build backend services."


def test_map_job_with_non_annual_salary_drops_salary(raw_jobs: list[dict]) -> None:
    job = LeverAdapter().map(raw_jobs[1])

    # per-hour-salary isn't comparable in our period-less columns -> unknown
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None
    assert job.location == "Remote"  # falls back to categories.location
    assert job.job_type == "Contract"


def test_remote_jobs_keeps_only_remote_workplace_type() -> None:
    postings = [
        {"id": "1", "workplaceType": "remote"},
        {"id": "2", "workplaceType": "hybrid"},
        {"id": "3", "workplaceType": "onsite"},
        {"id": "4"},  # no workplaceType
    ]
    assert LeverAdapter._remote_jobs(postings) == [{"id": "1", "workplaceType": "remote"}]


@pytest.mark.parametrize(
    ("salary_range", "expected"),
    [
        (
            {"min": 120000, "max": 160000, "currency": "USD", "interval": "per-year-salary"},
            (120000, 160000, "USD"),
        ),
        (
            {"min": 80, "max": 100, "currency": "USD", "interval": "per-hour-salary"},
            (None, None, None),
        ),
        (
            {"min": None, "max": None, "currency": "USD", "interval": "per-year-salary"},
            (None, None, None),
        ),
        (None, (None, None, None)),
    ],
)
def test_salary_keeps_only_annual(salary_range: dict | None, expected: tuple) -> None:
    assert LeverAdapter._salary(salary_range) == expected
