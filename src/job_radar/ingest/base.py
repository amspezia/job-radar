from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

# Sent on every outbound adapter request so sources can identify the client.
USER_AGENT = "job-radar/0.1"


@dataclass
class NormalizedJob:
    source: str
    source_type: str
    source_id: str | None
    url: str
    title: str
    company: str
    description: str
    salary_min: int | None
    salary_max: int | None
    currency: str | None
    location: str | None
    job_type: str | None
    remote: bool
    published_at: datetime | None


class SourceAdapter(ABC):
    source: str
    source_type: str

    @abstractmethod
    async def fetch(self) -> list[dict]: ...

    @abstractmethod
    def map(self, raw: dict) -> NormalizedJob: ...
