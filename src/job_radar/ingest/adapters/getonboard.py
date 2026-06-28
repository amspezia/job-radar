import logging
from datetime import UTC, datetime

import httpx

from job_radar.ingest.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.ingest.normalize import html_to_text

_API = "https://www.getonbrd.com/api/v0"
_PER_PAGE = 50
_MAX_PAGES_PER_CATEGORY = 8  # ~400 most-recent results per category.
# GetOnBoard category slugs — a TARGETING decision; add more dev categories
# (e.g. "sysadmin-devops-qa", "data-science-analytics") as needed.
_CATEGORIES = ["programming"]

logger = logging.getLogger(__name__)


class GetOnBoardAdapter(SourceAdapter):
    source = "getonboard"
    source_type = "aggregator"

    async def fetch(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            jobs: list[dict] = []
            for category in _CATEGORIES:
                jobs.extend(await self._fetch_category(client, category))
            return [j for j in jobs if j["attributes"].get("remote")]

    async def _fetch_category(self, client: httpx.AsyncClient, category: str) -> list[dict]:
        # expand[]=company embeds the full company object inline (incl. name),
        # avoiding a per-company follow-up request — which previously got us
        # rate-limited (429) once a page had many distinct companies.
        results: list[dict] = []
        for page in range(1, _MAX_PAGES_PER_CATEGORY + 1):
            resp = await client.get(
                f"{_API}/categories/{category}/jobs",
                params={"page": page, "per_page": _PER_PAGE, "expand[]": "company"},
            )
            resp.raise_for_status()
            payload = resp.json()
            results.extend(payload["data"])
            if page >= payload["meta"]["total_pages"]:
                break
        return results

    @staticmethod
    def _company_name(attrs: dict) -> str:
        company = (attrs.get("company") or {}).get("data") or {}
        return (company.get("attributes") or {}).get("name") or "Unknown"

    @staticmethod
    def _salary(attrs: dict) -> tuple[int | None, int | None, str | None]:
        # GetOnBoard salaries are gross MONTHLY USD; annualize for comparability
        # with the other sources' annual figures.
        min_monthly, max_monthly = attrs.get("min_salary"), attrs.get("max_salary")
        if not min_monthly and not max_monthly:
            return None, None, None
        return (
            min_monthly * 12 if min_monthly else None,
            max_monthly * 12 if max_monthly else None,
            "USD",
        )

    def map(self, raw: dict) -> NormalizedJob:
        attrs = raw["attributes"]
        salary_min, salary_max, currency = self._salary(attrs)
        published = attrs.get("published_at")
        countries = attrs.get("countries") or []
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=raw["id"],
            url=raw["links"]["public_url"],
            title=attrs["title"],
            company=self._company_name(attrs),
            description=html_to_text(attrs.get("description") or ""),
            salary_min=salary_min,
            salary_max=salary_max,
            currency=currency,
            location=attrs.get("remote_zone") or (", ".join(countries) or None),
            job_type=None,
            remote=True,  # fetch() already filtered to remote-only postings.
            published_at=datetime.fromtimestamp(published, tz=UTC) if published else None,
        )
