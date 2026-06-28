from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


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
