import json
import re
from pathlib import Path

import httpx
import pytest

from job_radar.ingest import discovery

_GREENHOUSE_RE = re.compile(r"(?:boards|job-boards)\.greenhouse\.io/([a-zA-Z0-9_-]+)")


def test_parse_tokens_filters_tech_and_matching_ats() -> None:
    companies = [
        {"industry_category": "tech", "ats_links": ["https://boards.greenhouse.io/GitLab"]},
        {
            "industry_category": "tech",
            "ats_links": ["https://job-boards.greenhouse.io/databricks/jobs"],
        },
        {"industry_category": "gaming", "ats_links": ["https://boards.greenhouse.io/somegame"]},
        {"industry_category": "tech", "ats_links": ["https://jobs.lever.co/foo"]},
        {"industry_category": "tech", "ats_links": ["https://boards.greenhouse.io/gitlab"]},
        {"industry_category": "tech"},
    ]
    tokens = discovery._parse_tokens(companies, _GREENHOUSE_RE)

    # gaming excluded, lever excluded, "GitLab"/"gitlab" deduped to lowercase
    assert tokens == {"gitlab", "databricks"}


async def test_get_tokens_returns_and_caches_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_discover(client: object, link_regex: object, board_url: object) -> list[str]:
        return ["alpha", "beta"]

    monkeypatch.setattr(discovery, "_discover", fake_discover)
    cache = tmp_path / "tokens.json"

    result = await discovery.get_tokens(
        None, link_regex=_GREENHOUSE_RE, board_url="u/{token}", cache_path=cache
    )

    assert result == ["alpha", "beta"]
    assert json.loads(cache.read_text()) == ["alpha", "beta"]


async def test_get_tokens_falls_back_to_cache_on_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "tokens.json"
    cache.write_text(json.dumps(["cached"]))

    async def boom(client: object, link_regex: object, board_url: object) -> list[str]:
        raise httpx.HTTPError("openjobs unreachable")

    monkeypatch.setattr(discovery, "_discover", boom)

    result = await discovery.get_tokens(
        None, link_regex=_GREENHOUSE_RE, board_url="u/{token}", cache_path=cache
    )

    assert result == ["cached"]


async def test_get_tokens_falls_back_to_cache_when_discovery_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "tokens.json"
    cache.write_text(json.dumps(["cached"]))

    async def empty(client: object, link_regex: object, board_url: object) -> list[str]:
        return []

    monkeypatch.setattr(discovery, "_discover", empty)

    result = await discovery.get_tokens(
        None, link_regex=_GREENHOUSE_RE, board_url="u/{token}", cache_path=cache
    )

    assert result == ["cached"]


def test_load_cache_missing_file_returns_empty(tmp_path: Path) -> None:
    assert discovery._load_cache(tmp_path / "nope.json") == []
