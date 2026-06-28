import json
from datetime import datetime
from pathlib import Path

import pytest

from job_radar.ingest.adapters.remotive import RemotiveAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "remotive_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_salaried_job(raw_jobs: list[dict]) -> None:
    job = RemotiveAdapter().map(raw_jobs[0])

    assert job.source == "remotive"
    assert job.source_type == "aggregator"
    assert job.source_id == "2091035"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme Cloud"
    assert job.job_type == "full_time"
    assert job.location == "Worldwide"
    assert job.remote is True
    assert job.published_at == datetime(2026, 6, 24, 11, 41, 57)

    # salary parsed from "$80k - $120k"
    assert job.salary_min == 80000
    assert job.salary_max == 120000
    assert job.currency == "USD"

    # HTML stripped to clean, space-separated text
    assert "<" not in job.description
    assert job.description == "We are hiring a backend engineer . Python Postgres"


def test_map_job_without_salary(raw_jobs: list[dict]) -> None:
    job = RemotiveAdapter().map(raw_jobs[1])

    # empty salary string -> unknown, never fabricated
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None
    assert job.job_type == "contract"
    assert job.description == "Own the product roadmap."
