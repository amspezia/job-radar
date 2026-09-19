"""What a judge is given, what it returns, and how it fails.

The brief and the few-shot examples are fixed when a judge is constructed, so the
static part of its prompt is identical for every job it grades (design §2.4: the
prefix must be byte-identical for Ollama's prefix cache to hit).
"""

from dataclasses import dataclass
from typing import Protocol


class JudgeError(RuntimeError):
    """A judge returned nothing usable for a job.

    The runner retries and counts these; a judge never invents a default grade.
    """


@dataclass(frozen=True)
class CandidateBrief:
    """The candidate side of the judgment — the profile snapshot, judge-ready."""

    seniority: str
    years_experience: float | None
    target_titles: list[str]
    tech_stack: list[str]
    domains: list[str]
    remote_required: bool
    cv_text: str

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "CandidateBrief":
        """Read a topic's `profile_snapshot`, tolerating absent optional keys."""
        return cls(
            seniority=snapshot.get("seniority") or "unknown",
            years_experience=snapshot.get("years_experience"),
            target_titles=list(snapshot.get("target_titles") or []),
            tech_stack=list(snapshot.get("tech_stack") or []),
            domains=list(snapshot.get("domains") or []),
            remote_required=bool(snapshot.get("remote_required")),
            cv_text=snapshot.get("cv_text") or "",
        )


@dataclass(frozen=True)
class JobView:
    """The posting side — exactly the fields the human label view shows (F21)."""

    title: str
    company: str
    source: str
    seniority: str | None
    requirements: str | None
    responsibilities: str | None
    description: str


@dataclass
class Judgment:
    grade: int
    rationale: str
    prompt_eval_seconds: float | None = None


class Judge(Protocol):
    name: str

    def describe(self) -> dict: ...

    async def grade(self, job: JobView) -> Judgment: ...
