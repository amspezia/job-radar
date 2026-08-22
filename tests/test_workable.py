import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from job_radar.adapters.sources.workable import WorkableAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "workable_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_remote_job(raw_jobs: list[dict]) -> None:
    job = WorkableAdapter().map(raw_jobs[0])

    assert job.source == "workable"
    assert job.source_type == "board"
    assert job.source_id == "8174486B78"
    assert job.url == "https://apply.workable.com/j/8174486B78"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Acme Labs"  # account-level name, not the token
    assert job.job_type == "Full-time"
    assert job.remote is True

    # Workable's widget API exposes no salary.
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None

    # description is HTML and must be stripped
    assert "<" not in job.description
    assert job.description == "Build backend services."


def test_published_at_is_utc_aware(raw_jobs: list[dict]) -> None:
    # published_on is a bare date; published_at is compared against a tz-aware
    # cutoff in retrieval/filters.py, so a naive datetime would not compare.
    published_at = WorkableAdapter().map(raw_jobs[0]).published_at

    assert published_at == datetime(2026, 5, 5, tzinfo=UTC)
    assert published_at.tzinfo is not None


def test_map_handles_missing_published_on(raw_jobs: list[dict]) -> None:
    assert WorkableAdapter().map(raw_jobs[2]).published_at is None


def test_location_renders_city_region_country(raw_jobs: list[dict]) -> None:
    assert WorkableAdapter().map(raw_jobs[0]).location == "Los Angeles, California, United States"


def test_location_keeps_every_country_of_a_multi_country_role(raw_jobs: list[dict]) -> None:
    # The flat country field holds only "Brazil"; the locations array is the
    # only place the full eligible set survives.
    assert WorkableAdapter().map(raw_jobs[2]).location == "Brazil, Honduras, Uruguay"


def test_location_is_none_when_no_locations_given() -> None:
    assert WorkableAdapter._location({"locations": []}) is None


def test_remote_filter_keeps_only_telecommuting_postings(raw_jobs: list[dict]) -> None:
    kept = WorkableAdapter._remote_jobs(raw_jobs)

    assert [j["title"] for j in kept] == ["Senior Backend Engineer", "Site Reliability Engineer"]
