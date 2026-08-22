import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from job_radar.adapters.sources.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.adapters.sources.discovery import get_tokens
from job_radar.adapters.sources.normalize import html_to_text

_BOARD_URL = "https://apply.workable.com/api/v1/widget/accounts/{token}"
# Both domains front the same ATS and the same account slug.
_LINK_RE = re.compile(r"(?:apply|jobs)\.workable\.com/([a-zA-Z0-9_-]+)")
_TOKENS_CACHE = Path("data/workable_tokens.json")

logger = logging.getLogger(__name__)


def _has_jobs(payload: dict) -> bool:
    # Workable nests postings under a "jobs" key, alongside the account name.
    return bool(payload.get("jobs"))


class WorkableAdapter(SourceAdapter):
    source = "workable"
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
                logger.warning("No Workable tokens available; skipping source.")
                return []
            for token in tokens:
                try:
                    # Without details=true the payload carries no description at
                    # all — same shape as Greenhouse's content=true, so the token
                    # discovery above can keep using the cheaper bare request.
                    resp = await client.get(
                        _BOARD_URL.format(token=token), params={"details": "true"}
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning("Skipping Workable board '%s': %s", token, exc)
                    continue
                payload = resp.json()
                company = payload.get("name")
                for posting in self._remote_jobs(payload.get("jobs") or []):
                    # Company lives on the account, not the posting.
                    posting["_company"] = company
                    jobs.append(posting)
        return jobs

    @staticmethod
    def _remote_jobs(postings: list[dict]) -> list[dict]:
        return [p for p in postings if p.get("telecommuting")]

    @staticmethod
    def _location(raw: dict) -> str | None:
        """Render `locations` — the only field that survives multi-country roles.

        The flat country/city/state fields agree with `locations[0]` everywhere
        they are populated (416/416 real postings checked), but they hold a
        single location, so a remote role open in four countries reports only
        the first. `locations` keeps the whole eligible set.
        """
        parts: list[str] = []
        for entry in raw.get("locations") or []:
            fields = [entry.get("city"), entry.get("region"), entry.get("country")]
            rendered = ", ".join(f for f in fields if f)
            if rendered and rendered not in parts:
                parts.append(rendered)
        return ", ".join(parts) or None

    def map(self, raw: dict) -> NormalizedJob:
        published = raw.get("published_on")
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=raw["shortcode"],
            url=raw["url"],
            title=raw["title"],
            company=raw["_company"],
            description=html_to_text(raw.get("description") or ""),
            salary_min=None,  # Workable's widget API exposes no salary.
            salary_max=None,
            currency=None,
            location=self._location(raw),
            job_type=raw.get("employment_type") or None,
            remote=True,  # fetch() already filtered to remote-only postings.
            # A bare "YYYY-MM-DD", so fromisoformat yields a naive datetime;
            # published_at is timestamptz and is compared against an aware
            # cutoff in retrieval/filters.py, so pin it to UTC explicitly.
            published_at=(
                datetime.fromisoformat(published).replace(tzinfo=UTC) if published else None
            ),
        )
