import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from job_radar.adapters.sources.arbeitnow import ArbeitnowAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "arbeitnow_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_remote_job(raw_jobs: list[dict]) -> None:
    job = ArbeitnowAdapter().map(raw_jobs[0])

    assert job.source == "arbeitnow"
    assert job.source_type == "aggregator"
    assert job.source_id == "senior-python-engineer-acme-279000"
    assert job.title == "Senior Python Engineer"
    assert job.company == "Acme GmbH"
    assert job.location == "Berlin"
    assert job.remote is True
    assert job.job_type == "Full Time"  # first of job_types
    assert job.published_at == datetime(2026, 6, 27, 23, 0, 32, tzinfo=UTC)

    # Arbeitnow exposes no salary.
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None

    assert "<" not in job.description
    assert job.description == "Join our backend team. Python Django"


def test_map_non_remote_job_with_empty_fields(raw_jobs: list[dict]) -> None:
    job = ArbeitnowAdapter().map(raw_jobs[1])

    assert job.remote is False
    assert job.location is None  # empty string -> None
    assert job.job_type is None  # empty job_types -> None
