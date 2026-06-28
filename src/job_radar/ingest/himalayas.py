import asyncio
from datetime import UTC, datetime

import httpx

from job_radar.ingest.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.ingest.normalize import html_to_text

_SEARCH_URL = "https://himalayas.app/jobs/api/search"
_MAX_PAGES_PER_QUERY = 6  # ~120 most-recent results per query (6 pages x 20).
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_BACKOFF = 5.0  # seconds, multiplied by the attempt number.

# Role keywords driving the search. This is a TARGETING decision — trim/extend
# freely. Use specific role *phrases* rather than bare words: the search is
# relevance-ranked, so "software engineer" returns more (and far cleaner)
# results than "software" or "engineer" alone. Cross-query duplicates collapse
# on content_hash, so overlap between keywords is harmless.
_QUERIES = [
    "software engineer",
    "backend developer",
    "frontend developer",
    "full stack developer",
    "mobile developer",
    "data engineer",
    "data scientist",
    "machine learning engineer",
    "devops engineer",
    "platform engineer",
    "security engineer",
    "qa engineer",
    "python developer",
    "react developer",
    "engineering manager",
]


class HimalayasAdapter(SourceAdapter):
    source = "himalayas"
    source_type = "aggregator"

    async def fetch(self) -> list[dict]:
        jobs: list[dict] = []
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            for query in _QUERIES:
                jobs.extend(await self._fetch_query(client, query))
        return jobs

    async def _fetch_query(self, client: httpx.AsyncClient, query: str) -> list[dict]:
        results: list[dict] = []
        for page in range(1, _MAX_PAGES_PER_QUERY + 1):
            payload = await self._get_page(client, query, page)
            batch = payload["jobs"]
            results.extend(batch)
            if not batch or payload["offset"] + len(batch) >= payload["totalCount"]:
                break
        return results

    @staticmethod
    async def _get_page(client: httpx.AsyncClient, query: str, page: int) -> dict:
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            resp = await client.get(_SEARCH_URL, params={"q": query, "page": page})
            if resp.status_code == 429 and attempt < _RATE_LIMIT_RETRIES:
                await asyncio.sleep(_RATE_LIMIT_BACKOFF * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")  # loop either returns or raises

    @staticmethod
    def _location(locations: list[str]) -> str | None:
        if not locations:
            return None
        joined = ", ".join(locations)
        # Some postings list dozens of countries (effectively "anywhere"); the
        # joined string can exceed the location column's width. Rather than
        # truncate mid-name, collapse a long list to its honest meaning.
        if len(joined) > 255:
            return "Worldwide"
        return joined

    @staticmethod
    def _salary(raw: dict) -> tuple[int | None, int | None, str | None]:
        # Only keep annual figures; hourly/monthly aren't comparable in our
        # period-less salary columns, so they're stored as unknown.
        if raw.get("salaryPeriod") != "annual":
            return None, None, None
        min_salary, max_salary = raw.get("minSalary"), raw.get("maxSalary")
        if min_salary is None and max_salary is None:
            return None, None, None
        return min_salary, max_salary, raw.get("currency")

    def map(self, raw: dict) -> NormalizedJob:
        salary_min, salary_max, currency = self._salary(raw)
        published = raw.get("pubDate")
        locations = raw.get("locationRestrictions") or []
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=raw["guid"],
            url=raw["applicationLink"],
            title=raw["title"],
            company=raw["companyName"],
            description=html_to_text(raw.get("description", "")),
            salary_min=salary_min,
            salary_max=salary_max,
            currency=currency,
            location=self._location(locations),
            job_type=raw.get("employmentType") or None,
            remote=True,  # Himalayas is a remote-only board.
            published_at=datetime.fromtimestamp(published, tz=UTC) if published else None,
        )
