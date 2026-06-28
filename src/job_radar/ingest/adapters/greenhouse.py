import html
import logging
import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from job_radar.ingest.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.ingest.discovery import get_tokens
from job_radar.ingest.normalize import ParsedSalary, html_to_text, parse_salary

_BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
_LINK_RE = re.compile(r"(?:boards|job-boards)\.greenhouse\.io/([a-zA-Z0-9_-]+)")
_TOKENS_CACHE = Path("data/greenhouse_tokens.json")

logger = logging.getLogger(__name__)


class GreenHouseAdapter(SourceAdapter):
    source = "greenhouse"
    source_type = "board"

    async def fetch(self) -> list[dict]:
        jobs: list[dict] = []
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            tokens = await get_tokens(
                client, link_regex=_LINK_RE, board_url=_BOARD_URL, cache_path=_TOKENS_CACHE
            )
            if not tokens:
                logger.warning("No Greenhouse tokens available; skipping source.")
                return []
            for token in tokens:
                try:
                    resp = await client.get(
                        _BOARD_URL.format(token=token), params={"content": "true"}
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("Skipping Greenhouse board '%s': %s", token, exc)
                    continue
                jobs.extend(self._remote_jobs(resp.json()["jobs"]))
        return jobs

    @staticmethod
    def _remote_jobs(jobs: list[dict]) -> list[dict]:
        return [
            job
            for job in jobs
            if "remote" in ((job.get("location") or {}).get("name") or "").lower()
        ]

    @staticmethod
    def _salary(content_html: str) -> ParsedSalary:
        # Greenhouse renders pay-transparency ranges in a `pay-range` element,
        # e.g. "<div class='pay-range'><span>$108,400</span>—<span>$129,600</span></div>".
        # Parse only that element, not the whole description, to avoid picking
        # up bonus/benefit figures. `content_html` must already be unescaped.
        pay_range = BeautifulSoup(content_html, "html.parser").select_one(".pay-range")
        if pay_range is None:
            return ParsedSalary(None, None, None)
        return parse_salary(pay_range.get_text(" "))

    def map(self, raw: dict) -> NormalizedJob:
        # Greenhouse `content` is HTML-*escaped* (e.g. "&lt;p&gt;"), so unescape
        # to real HTML once, then derive both the description and the salary.
        content = html.unescape(raw.get("content") or "")
        salary = self._salary(content)
        published = raw.get("first_published")
        location = (raw.get("location") or {}).get("name")
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=str(raw["id"]),
            url=raw["absolute_url"],
            title=raw["title"],
            company=raw["company_name"],
            description=html_to_text(content),
            salary_min=salary.min,
            salary_max=salary.max,
            currency=salary.currency,
            location=location,
            job_type=None,  # no employment-type field on the board API.
            remote=True,  # fetch() already filtered to remote-only postings.
            published_at=datetime.fromisoformat(published) if published else None,
        )
