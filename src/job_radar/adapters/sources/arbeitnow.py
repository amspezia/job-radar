from datetime import UTC, datetime

import httpx

from job_radar.adapters.sources.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.adapters.sources.normalize import html_to_text

_API_URL = "https://www.arbeitnow.com/api/job-board-api"
# Bounds volume (and embedding cost) per run; Arbeitnow serves 100 jobs/page.
_MAX_PAGES = 5


class ArbeitnowAdapter(SourceAdapter):
    source = "arbeitnow"
    source_type = "aggregator"

    async def fetch(self) -> list[dict]:
        # Paginate by explicit page number rather than following the API's
        # `links.next`: Arbeitnow bakes a random featured search into each next
        # link (e.g. ?search=...&location=spain), which makes pagination
        # non-deterministic and biases pages toward narrow filtered slices.
        jobs: list[dict] = []
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            for page in range(1, _MAX_PAGES + 1):
                resp = await client.get(_API_URL, params={"page": page})
                resp.raise_for_status()
                data = resp.json()["data"]
                if not data:
                    break
                jobs.extend(data)
        return [job for job in jobs if job.get("remote")]

    def map(self, raw: dict) -> NormalizedJob:
        created = raw.get("created_at")
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=raw["slug"],
            url=raw["url"],
            title=raw["title"],
            company=raw["company_name"],
            description=html_to_text(raw.get("description", "")),
            salary_min=None,  # Arbeitnow does not expose salary.
            salary_max=None,
            currency=None,
            location=raw.get("location") or None,
            job_type=_first_job_type(raw),
            remote=raw["remote"],
            published_at=datetime.fromtimestamp(created, tz=UTC) if created else None,
        )


def _first_job_type(raw: dict) -> str | None:
    """Arbeitnow's `job_types` is normally a JSON array (e.g. ["Full Time"]), but
    for postings whose underlying array isn't sequentially indexed from 0, their
    API serializes it as an object instead (e.g. {"1": "professional / experienced"})
    — same data, different JSON shape depending on an implementation detail on
    their end. Handle both rather than assuming list indexing always works.
    """
    job_types = raw.get("job_types") or []
    if isinstance(job_types, dict):
        return next(iter(job_types.values()), None)
    return job_types[0] if job_types else None
