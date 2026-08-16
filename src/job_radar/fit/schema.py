from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A verbatim quote backing a judgment, tagged with where it came from.

    Verbatim (not paraphrased) so a later guardrail can verify the quote
    actually appears in its source.
    """

    source: Literal["profile", "posting"]
    quote: str


class Requirement(BaseModel):
    """One posting requirement and how well the candidate meets it.

    Deliberately carries no paraphrase of the requirement: the `posting` evidence
    quote identifies it verbatim, which is both cheaper to decode and stronger for
    audit than a restatement. Decode is ~93% of fit-run wall time, so every field
    here is paid for one forward pass per token.

    **Field order is generation order** — constrained decoding emits these in the
    sequence declared here, so `evidence` comes first deliberately: the model must
    surface the quotes before committing to a classification. Measured, not
    stylistic. Judging first and quoting afterwards made the model markedly
    harsher (whole postings flipping to all-"unmet"), because it then had nothing
    to reason over when it picked the label.
    """

    evidence: list[Evidence]
    kind: Literal["required", "preferred"]
    satisfaction: Literal["met", "partial", "unmet"]


class DomainJudgment(BaseModel):
    relevance: Literal["strong", "partial", "weak"]
    evidence: list[Evidence]


class FitJudgment(BaseModel):
    """The grounded classifications the LLM returns — deliberately no score.

    The score is computed deterministically from this by fit.score, so the
    model can never hand us an unverifiable number. Seniority is not judged here
    — it is structured metadata on the posting (Job.seniority).
    """

    requirements: list[Requirement]
    domain: DomainJudgment
    # The prompt asks for one sentence; this bound is a backstop against a runaway
    # draft, not the target. Kept well above the asked-for length deliberately —
    # a tight bound would turn an over-long summary into a hard validation failure
    # that discards the whole assessment, trading 70 tokens for the entire job.
    summary: str = Field(max_length=400)


class FitAssessment(BaseModel):
    """The final result code returns to callers."""

    score: int | None  # 0-100; None when pre-flight refused (insufficient input)
    verdict: Literal["strong", "moderate", "weak", "none"]
    gate_failed: bool
    judgment: FitJudgment | None  # the grounded evidence behind the score
    summary: str
