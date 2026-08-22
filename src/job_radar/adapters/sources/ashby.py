import logging
import re
from datetime import datetime
from pathlib import Path

import httpx

from job_radar.adapters.sources.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.adapters.sources.discovery import get_tokens
from job_radar.adapters.sources.normalize import html_to_text

_BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}"
_LINK_RE = re.compile(r"jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)")
_TOKENS_CACHE = Path("data/ashby_tokens.json")

logger = logging.getLogger(__name__)


def _has_jobs(payload: dict) -> bool:
    # Ashby nests postings under a "jobs" key.
    return bool(payload.get("jobs"))


class AshbyAdapter(SourceAdapter):
    source = "ashby"
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
                logger.warning("No Ashby tokens available; skipping source.")
                return []
            for token in tokens:
                try:
                    resp = await client.get(_BOARD_URL.format(token=token))
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("Skipping Ashby board '%s': %s", token, exc)
                    continue
                for posting in self._remote_jobs(resp.json().get("jobs") or []):
                    # Ashby exposes no company name anywhere in the payload
                    # (checked across several live boards); the board token is
                    # the company slug, so stash it for map() — same approach
                    # lever.py already uses for the same gap.
                    posting["_token"] = token
                    jobs.append(posting)
        return jobs

    @staticmethod
    def _remote_jobs(postings: list[dict]) -> list[dict]:
        # isListed guards against boards exposing drafts/unlisted reqs.
        return [p for p in postings if p.get("isRemote") and p.get("isListed", True)]

    def map(self, raw: dict) -> NormalizedJob:
        published = raw.get("publishedAt")
        # descriptionPlain is already plain text; only fall back to stripping the
        # HTML variant when it is absent, so the common path does no parsing.
        description = raw.get("descriptionPlain") or html_to_text(raw.get("descriptionHtml") or "")
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=raw["id"],
            url=raw["jobUrl"],
            title=raw["title"],
            company=raw["_token"].replace("-", " ").replace("_", " ").title(),
            description=description,
            salary_min=None,  # Ashby's public board API exposes no salary.
            salary_max=None,
            currency=None,
            location=raw.get("location") or None,
            job_type=raw.get("employmentType") or None,
            remote=True,  # fetch() already filtered to remote-only postings.
            published_at=datetime.fromisoformat(published) if published else None,
        )
