from pydantic import BaseModel


class WorkItem(BaseModel):
    role: str
    company: str | None
    years: float | None  # computed from start/end by parse_cv, not model-sourced
    start: str | None  # as written on the CV, e.g. "May 2020"
    end: str | None  # as written on the CV, e.g. "Feb 2024" or "Present"
    highlights: list[str]  # notable achievements / responsibilities, verbatim


class StructuredProfile(BaseModel):
    full_name: str | None  # PII — stored locally only
    email: str | None  # PII — stored locally only
    seniority: str | None
    target_titles: list[str]
    tech_stack: list[str]
    domains: list[str]
    work_history: list[WorkItem]
    years_experience: float | None  # computed from work_history by parse_cv, not model-sourced
