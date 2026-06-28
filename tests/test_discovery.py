import asyncio
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


def _truthy(payload: object) -> bool:
    return bool(payload)


async def test_get_tokens_returns_and_caches_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_discover(
        client: object, link_regex: object, board_url: object, has_jobs: object
    ) -> list[str]:
        return ["alpha", "beta"]

    monkeypatch.setattr(discovery, "_discover", fake_discover)
    cache = tmp_path / "tokens.json"

    result = await discovery.get_tokens(
        None, link_regex=_GREENHOUSE_RE, board_url="u/{token}", cache_path=cache, has_jobs=_truthy
    )

    assert result == ["alpha", "beta"]
    assert json.loads(cache.read_text()) == ["alpha", "beta"]


async def test_get_tokens_falls_back_to_cache_on_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "tokens.json"
    cache.write_text(json.dumps(["cached"]))

    async def boom(
        client: object, link_regex: object, board_url: object, has_jobs: object
    ) -> list[str]:
        raise httpx.HTTPError("openjobs unreachable")

    monkeypatch.setattr(discovery, "_discover", boom)

    result = await discovery.get_tokens(
        None, link_regex=_GREENHOUSE_RE, board_url="u/{token}", cache_path=cache, has_jobs=_truthy
    )

    assert result == ["cached"]


async def test_get_tokens_falls_back_to_cache_when_discovery_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "tokens.json"
    cache.write_text(json.dumps(["cached"]))

    async def empty(
        client: object, link_regex: object, board_url: object, has_jobs: object
    ) -> list[str]:
        return []

    monkeypatch.setattr(discovery, "_discover", empty)

    result = await discovery.get_tokens(
        None, link_regex=_GREENHOUSE_RE, board_url="u/{token}", cache_path=cache, has_jobs=_truthy
    )

    assert result == ["cached"]


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeClient:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.requested: str | None = None

    async def get(self, url: str) -> _FakeResponse:
        self.requested = url
        return _FakeResponse(200, self._payload)


async def test_verify_uses_caller_predicate_for_payload_shape() -> None:
    sem = asyncio.Semaphore(1)

    # Greenhouse-style payload: jobs nested under a key.
    gh_client = _FakeClient({"jobs": [{"id": 1}]})
    gh = await discovery._verify(gh_client, sem, "acme", "u/{token}", lambda p: bool(p.get("jobs")))
    assert gh == "acme"
    assert gh_client.requested == "u/acme"

    # Lever-style payload: a bare list.
    lever = await discovery._verify(
        _FakeClient([{"id": "x"}]), sem, "acme", "u/{token}", lambda p: bool(p)
    )
    assert lever == "acme"

    # Empty board rejected regardless of shape.
    empty = await discovery._verify(_FakeClient([]), sem, "acme", "u/{token}", lambda p: bool(p))
    assert empty is None


def test_load_cache_missing_file_returns_empty(tmp_path: Path) -> None:
    assert discovery._load_cache(tmp_path / "nope.json") == []
