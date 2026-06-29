import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# OpenJobs is a community dataset mapping companies to their ATS links, tagged by
# industry. The same source yields tokens for every ATS (greenhouse/lever/ashby);
# callers pass the per-ATS link regex, board URL, and a predicate that reports
# whether a board's payload contains jobs — everything else is shared.
_OPENJOBS_URL = "https://raw.githubusercontent.com/outscal/OpenJobs/main/data/companies_v2.json"
_RELEVANT_CATEGORY = "tech"
_CONCURRENCY = 15

# Reports whether a board's verify-response payload holds any published jobs.
# ATS payload shapes differ (Greenhouse nests under "jobs", Lever returns a bare
# list), so each adapter supplies its own predicate.
HasJobs = Callable[[Any], bool]


async def _fetch_companies(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(_OPENJOBS_URL)
    resp.raise_for_status()
    return resp.json()


def _parse_tokens(companies: list[dict], link_regex: re.Pattern[str]) -> set[str]:
    tokens: set[str] = set()
    for company in companies:
        if company.get("industry_category") != _RELEVANT_CATEGORY:
            continue
        for link in company.get("ats_links") or []:
            match = link_regex.search(link)
            if match:
                tokens.add(match.group(1).lower())
    return tokens


async def _verify(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    token: str,
    board_url: str,
    has_jobs: HasJobs,
) -> str | None:
    # Lightweight liveness check (no ?content=true): the token is real if the
    # board resolves and the caller's predicate sees published jobs in it.
    async with sem:
        try:
            resp = await client.get(board_url.format(token=token))
            if resp.status_code == 200 and has_jobs(resp.json()):
                return token
        except httpx.HTTPError:
            return None
    return None


async def _discover(
    client: httpx.AsyncClient, link_regex: re.Pattern[str], board_url: str, has_jobs: HasJobs
) -> list[str]:
    companies = await _fetch_companies(client)
    tokens = _parse_tokens(companies, link_regex)
    sem = asyncio.Semaphore(_CONCURRENCY)
    verified = await asyncio.gather(*(_verify(client, sem, t, board_url, has_jobs) for t in tokens))
    return sorted(token for token in verified if token)


def _load_cache(path: Path) -> list[str]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _save_cache(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokens, indent=2))


async def get_tokens(
    client: httpx.AsyncClient,
    *,
    link_regex: re.Pattern[str],
    board_url: str,
    cache_path: Path,
    has_jobs: HasJobs,
) -> list[str]:
    """Discover live ATS board tokens, caching the result.

    On a successful discovery the cache is refreshed and returned. If discovery
    fails (e.g. OpenJobs unreachable) or yields nothing, fall back to the last
    cached list so a transient outage doesn't drop the source entirely.
    """
    try:
        tokens = await _discover(client, link_regex, board_url, has_jobs)
    except httpx.HTTPError as exc:
        logger.warning("Token discovery failed (%s); falling back to cache %s", exc, cache_path)
        return _load_cache(cache_path)

    if tokens:
        _save_cache(cache_path, tokens)
        return tokens

    logger.warning("Token discovery returned nothing; falling back to cache %s", cache_path)
    return _load_cache(cache_path)
