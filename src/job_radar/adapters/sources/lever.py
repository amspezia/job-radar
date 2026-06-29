import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from job_radar.adapters.sources.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.adapters.sources.discovery import get_tokens
from job_radar.adapters.sources.normalize import html_to_text

# mode=json returns the full postings list as a bare JSON array (not nested).
_BOARD_URL = "https://api.lever.co/v0/postings/{token}?mode=json"
_LINK_RE = re.compile(r"jobs\.lever\.co/([a-zA-Z0-9_-]+)")
_TOKENS_CACHE = Path("data/lever_tokens.json")

logger = logging.getLogger(__name__)


def _has_jobs(payload: list) -> bool:
    # Lever returns a bare list of postings, not a dict.
    return bool(payload)


class LeverAdapter(SourceAdapter):
    source = "lever"
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
                logger.warning("No Lever tokens available; skipping source.")
                return []
            for token in tokens:
                try:
                    resp = await client.get(_BOARD_URL.format(token=token))
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("Skipping Lever board '%s': %s", token, exc)
                    continue
                for posting in self._remote_jobs(resp.json()):
                    # Lever postings carry no company name; the board token is
                    # the company slug, so stash it for map() to derive one.
                    posting["_token"] = token
                    jobs.append(posting)
        return jobs

    @staticmethod
    def _remote_jobs(postings: list[dict]) -> list[dict]:
        return [p for p in postings if p.get("workplaceType") == "remote"]

    @staticmethod
    def _salary(salary_range: dict | None) -> tuple[int | None, int | None, str | None]:
        # Lever exposes structured ranges; keep only annual figures so they're
        # comparable with the other sources' period-less salary columns.
        if not salary_range or salary_range.get("interval") != "per-year-salary":
            return None, None, None
        min_salary, max_salary = salary_range.get("min"), salary_range.get("max")
        if not min_salary and not max_salary:
            return None, None, None
        return min_salary, max_salary, salary_range.get("currency")

    def map(self, raw: dict) -> NormalizedJob:
        categories = raw.get("categories") or {}
        salary_min, salary_max, currency = self._salary(raw.get("salaryRange"))
        created = raw.get("createdAt")  # epoch milliseconds
        locations = categories.get("allLocations") or (
            [categories["location"]] if categories.get("location") else []
        )
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=raw["id"],
            url=raw["hostedUrl"],
            title=raw["text"],
            company=raw["_token"].replace("-", " ").replace("_", " ").title(),
            description=html_to_text(raw.get("description") or ""),
            salary_min=salary_min,
            salary_max=salary_max,
            currency=currency,
            location=", ".join(locations) or None,
            job_type=categories.get("commitment") or None,
            remote=True,  # fetch() already filtered to remote-only postings.
            published_at=datetime.fromtimestamp(created / 1000, tz=UTC) if created else None,
        )
