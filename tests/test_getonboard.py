import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from job_radar.adapters.sources.getonboard import GetOnBoardAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "getonboard_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_job_with_salary(raw_jobs: list[dict]) -> None:
    job = GetOnBoardAdapter().map(raw_jobs[0])

    assert job.source == "getonboard"
    assert job.source_type == "aggregator"
    assert job.source_id == "senior-java-developer-acme-santiago"
    assert job.url == "https://www.getonbrd.com/jobs/senior-java-developer-acme-santiago"
    assert job.title == "Senior Java Developer"
    assert job.company == "Acme"
    assert job.location == "Remote"
    assert job.remote is True
    assert job.published_at == datetime.fromtimestamp(1782511711, tz=UTC)

    # monthly USD annualized: 2700/2900 * 12
    assert job.salary_min == 32400
    assert job.salary_max == 34800
    assert job.currency == "USD"

    assert "<" not in job.description
    assert job.description == "Build backend services."


def test_map_job_without_salary(raw_jobs: list[dict]) -> None:
    job = GetOnBoardAdapter().map(raw_jobs[1])

    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None
    assert job.location is None  # no remote_zone, empty countries
    assert job.company == "Globex"


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"min_salary": 2700, "max_salary": 2900}, (32400, 34800, "USD")),
        ({"min_salary": 2000, "max_salary": None}, (24000, None, "USD")),
        ({"min_salary": None, "max_salary": None}, (None, None, None)),
        ({"min_salary": 0, "max_salary": 0}, (None, None, None)),
        ({}, (None, None, None)),
    ],
)
def test_salary_annualizes_monthly_usd(attrs: dict, expected: tuple) -> None:
    assert GetOnBoardAdapter._salary(attrs) == expected
