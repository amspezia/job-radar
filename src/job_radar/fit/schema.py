from typing import Literal

from pydantic import BaseModel


class Evidence(BaseModel):
    """A verbatim quote backing a judgment, tagged with where it came from.

    Verbatim (not paraphrased) so a later guardrail can verify the quote
    actually appears in its source.
    """

    source: Literal["profile", "posting"]
    quote: str


class Requirement(BaseModel):
    text: str
    kind: Literal["required", "preferred"]
    is_gate: bool  # a knockout dealbreaker (work authorization, location, ...)
    satisfaction: Literal["met", "partial", "unmet"]
    evidence: list[Evidence]


class SeniorityJudgment(BaseModel):
    posting_level: str
    candidate_level: str
    alignment: Literal["exact", "adjacent", "mismatch"]
    evidence: list[Evidence]


class DomainJudgment(BaseModel):
    relevance: Literal["strong", "partial", "weak"]
    evidence: list[Evidence]


class FitJudgment(BaseModel):
    """The grounded classifications the LLM returns — deliberately no score.

    The score is computed deterministically from this by fit.score, so the
    model can never hand us an unverifiable number.
    """

    requirements: list[Requirement]
    seniority: SeniorityJudgment
    domain: DomainJudgment
    summary: str


class FitAssessment(BaseModel):
    """The final result code returns to callers."""

    score: int | None  # 0-100; None when pre-flight refused (insufficient input)
    verdict: Literal["strong", "moderate", "weak", "none"]
    gate_failed: bool
    judgment: FitJudgment | None  # the grounded evidence behind the score
    summary: str
