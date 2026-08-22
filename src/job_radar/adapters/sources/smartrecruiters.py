import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path

import httpx

from job_radar.adapters.sources.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.adapters.sources.discovery import get_tokens
from job_radar.adapters.sources.normalize import html_to_text

_BOARD_URL = "https://api.smartrecruiters.com/v1/companies/{token}/postings"
# Both domains front the same ATS and the same company identifier.
_LINK_RE = re.compile(r"(?:jobs|careers)\.smartrecruiters\.com/([a-zA-Z0-9_-]+)")
_TOKENS_CACHE = Path("data/smartrecruiters_tokens.json")

# The API caps a page at 100 server-side; asking for more is silently ignored.
_PAGE_SIZE = 100
# Pages are cheap, so there is no page cap: remote postings are scattered
# through the large boards rather than clustered at the front (one 4784-posting
# board holds its 72 remote roles across all 48 pages), so capping pages would
# quietly drop most of the yield from exactly the biggest employers.
#
# The real bound is the remote filter, which runs on list results *before* any
# detail request — measured over every live board, that is 1.6k detail calls
# instead of 15.7k. Those remaining calls run concurrently because sequentially
# they take ~1s each.
_DETAIL_CONCURRENCY = 10

# Company boilerplate and "additional information" are dropped: only the two
# sections that actually carry requirements reach the description, matching how
# the rest of the pipeline prefers requirement-bearing text over company copy.
_WANTED_SECTIONS = ("jobDescription", "qualifications")

logger = logging.getLogger(__name__)


def _has_jobs(payload: dict) -> bool:
    # SmartRecruiters returns postings under "content", with paging metadata.
    return bool(payload.get("content"))


class SmartRecruitersAdapter(SourceAdapter):
    source = "smartrecruiters"
    source_type = "board"

    async def fetch(self) -> list[dict]:
        jobs: list[dict] = []
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            tokens = await get_tokens(
                client,
                link_regex=_LINK_RE,
                board_url=_BOARD_URL,
                cache_path=_TOKENS_CACHE,
                has_jobs=_has_jobs,
            )
            if not tokens:
                logger.warning("No SmartRecruiters tokens available; skipping source.")
                return []
            semaphore = asyncio.Semaphore(_DETAIL_CONCURRENCY)
            for token in tokens:
                postings = await self._list_remote(client, token)
                if not postings:
                    continue
                details = await asyncio.gather(
                    *(self._detail(client, semaphore, p) for p in postings)
                )
                jobs.extend(d for d in details if d is not None)
        return jobs

    @staticmethod
    async def _list_remote(client: httpx.AsyncClient, token: str) -> list[dict]:
        """Page a company's postings, keeping only the remote ones.

        `location.remote` is present on every list item, so filtering here is
        what keeps the per-posting detail fetch below affordable.
        """
        remote: list[dict] = []
        offset = 0
        while True:
            try:
                resp = await client.get(
                    _BOARD_URL.format(token=token),
                    params={"offset": offset, "limit": _PAGE_SIZE},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Skipping SmartRecruiters board '%s': %s", token, exc)
                return remote
            content = resp.json().get("content") or []
            if not content:
                return remote
            remote.extend(p for p in content if (p.get("location") or {}).get("remote"))
            offset += len(content)

    @staticmethod
    async def _detail(
        client: httpx.AsyncClient, semaphore: asyncio.Semaphore, posting: dict
    ) -> dict | None:
        """Fetch one posting's full record; the list item carries its own URL.

        Returns the detail merged over the list item so `map()` keeps the single
        `raw: dict` signature every other adapter uses. None on failure, so one
        dead posting never drops the rest of the board.
        """
        async with semaphore:
            try:
                resp = await client.get(posting["ref"])
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Skipping SmartRecruiters posting '%s': %s", posting.get("id"), exc)
                return None
        return {**posting, **resp.json()}

    @staticmethod
    def _description(raw: dict) -> str:
        sections = ((raw.get("jobAd") or {}).get("sections")) or {}
        texts = [(sections.get(name) or {}).get("text") or "" for name in _WANTED_SECTIONS]
        return html_to_text(" ".join(t for t in texts if t))

    def map(self, raw: dict) -> NormalizedJob:
        published = raw.get("releasedDate")
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=str(raw["id"]),
            url=raw["postingUrl"],
            title=raw["name"],
            company=(raw.get("company") or {}).get("name") or "",
            description=self._description(raw),
            salary_min=None,  # No salary field is exposed on the public API.
            salary_max=None,
            currency=None,
            location=(raw.get("location") or {}).get("fullLocation") or None,
            job_type=(raw.get("typeOfEmployment") or {}).get("label") or None,
            remote=True,  # fetch() already filtered to remote-only postings.
            published_at=datetime.fromisoformat(published) if published else None,
        )
