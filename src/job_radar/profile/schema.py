from pydantic import BaseModel


class WorkItem(BaseModel):
    role: str
    company: str | None
    years: float | None


class StructuredProfile(BaseModel):
    full_name: str | None  # PII — stored locally only
    email: str | None  # PII — stored locally only
    seniority: str
    target_titles: list[str]
    tech_stack: list[str]
    domains: list[str]
    work_history: list[WorkItem]
    years_experience: float | None
