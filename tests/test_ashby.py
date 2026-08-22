import json
from datetime import datetime
from pathlib import Path

import pytest

from job_radar.adapters.sources.ashby import AshbyAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "ashby_jobs.json"


@pytest.fixture
def raw_jobs() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_map_remote_job(raw_jobs: list[dict]) -> None:
    job = AshbyAdapter().map(raw_jobs[0])

    assert job.source == "ashby"
    assert job.source_type == "board"
    assert job.source_id == "7f302222-2e06-4e37-8766-87b3bf6068e6"
    assert job.url == "https://jobs.ashbyhq.com/acme-labs/7f302222-2e06-4e37-8766-87b3bf6068e6"
    assert job.title == "Senior Backend Engineer"
    assert job.location == "Remote - US"
    assert job.job_type == "FullTime"
    assert job.remote is True
    assert job.published_at == datetime.fromisoformat("2026-03-16T09:00:46.953+00:00")

    # Ashby's public board API exposes no salary.
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.currency is None


def test_company_is_derived_from_the_board_token(raw_jobs: list[dict]) -> None:
    # Ashby exposes no company name anywhere in the payload, so the slug is
    # de-hyphenated and title-cased (same fallback lever.py uses).
    assert AshbyAdapter().map(raw_jobs[0]).company == "Acme Labs"


def test_description_uses_plain_text_without_stripping_html(raw_jobs: list[dict]) -> None:
    # descriptionPlain is already plain, so it is passed through untouched
    # rather than run back through an HTML parser.
    assert AshbyAdapter().map(raw_jobs[0]).description == "Build backend services."


def test_description_falls_back_to_html_when_plain_is_empty(raw_jobs: list[dict]) -> None:
    job = AshbyAdapter().map(raw_jobs[2])

    assert job.description == "Own the pipeline ."  # tags stripped from the HTML variant
    assert "<" not in job.description


def test_map_handles_missing_published_at(raw_jobs: list[dict]) -> None:
    assert AshbyAdapter().map(raw_jobs[1]).published_at is None


def test_remote_filter_keeps_only_remote_listed_postings(raw_jobs: list[dict]) -> None:
    kept = AshbyAdapter._remote_jobs(raw_jobs)

    assert [j["title"] for j in kept] == ["Senior Backend Engineer", "Data Engineer"]


def test_remote_filter_drops_unlisted_postings() -> None:
    unlisted = [{"title": "Draft Role", "isRemote": True, "isListed": False}]

    assert AshbyAdapter._remote_jobs(unlisted) == []
