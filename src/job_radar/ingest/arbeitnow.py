from datetime import UTC, datetime

import httpx

from job_radar.ingest.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.ingest.normalize import html_to_text

_API_URL = "https://www.arbeitnow.com/api/job-board-api"
# Bounds volume (and embedding cost) per run; Arbeitnow serves 100 jobs/page.
_MAX_PAGES = 5


class ArbeitnowAdapter(SourceAdapter):
    source = "arbeitnow"
    source_type = "aggregator"

    async def fetch(self) -> list[dict]:
        jobs: list[dict] = []
        url: str | None = _API_URL
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
            for _ in range(_MAX_PAGES):
                if not url:
                    break
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()
                jobs.extend(payload["data"])
                url = payload.get("links", {}).get("next")
        # The API's ?remote=true filter is unreliable, so filter client-side:
        # this is a remote-roles system, non-remote postings are noise.
        return [job for job in jobs if job.get("remote")]

    def map(self, raw: dict) -> NormalizedJob:
        created = raw.get("created_at")
        job_types = raw.get("job_types") or []
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
            job_type=job_types[0] if job_types else None,
            remote=raw["remote"],
            published_at=datetime.fromtimestamp(created, tz=UTC) if created else None,
        )
