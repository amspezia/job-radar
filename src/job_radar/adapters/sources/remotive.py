from datetime import datetime

import httpx

from job_radar.adapters.sources.base import USER_AGENT, NormalizedJob, SourceAdapter
from job_radar.adapters.sources.normalize import html_to_text, parse_salary

_API_URL = "https://remotive.com/api/remote-jobs"


class RemotiveAdapter(SourceAdapter):
    source = "remotive"
    source_type = "aggregator"

    async def fetch(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(_API_URL, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
        return resp.json()["jobs"]

    def map(self, raw: dict) -> NormalizedJob:
        salary = parse_salary(raw.get("salary", ""))
        published = raw.get("publication_date")
        return NormalizedJob(
            source=self.source,
            source_type=self.source_type,
            source_id=str(raw["id"]),
            url=raw["url"],
            title=raw["title"],
            company=raw["company_name"],
            description=html_to_text(raw.get("description", "")),
            salary_min=salary.min,
            salary_max=salary.max,
            currency=salary.currency,
            location=raw.get("candidate_required_location"),
            job_type=raw.get("job_type"),
            remote=True,
            published_at=datetime.fromisoformat(published) if published else None,
        )
