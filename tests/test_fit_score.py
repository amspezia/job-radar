import pytest

from job_radar.fit.schema import (
    DomainJudgment,
    FitJudgment,
    Requirement,
    SeniorityJudgment,
)
from job_radar.fit.score import _verdict, score_fit


def _req(
    kind: str = "required", satisfaction: str = "met", *, is_gate: bool = False
) -> Requirement:
    return Requirement(text="x", kind=kind, is_gate=is_gate, satisfaction=satisfaction, evidence=[])


def _judgment(
    requirements: list[Requirement], alignment: str = "exact", relevance: str = "strong"
) -> FitJudgment:
    return FitJudgment(
        requirements=requirements,
        seniority=SeniorityJudgment(
            posting_level="senior", candidate_level="senior", alignment=alignment, evidence=[]
        ),
        domain=DomainJudgment(relevance=relevance, evidence=[]),
        summary="ok",
    )


def test_all_dimensions_maxed_scores_100_strong() -> None:
    j = _judgment([_req("required"), _req("preferred")], alignment="exact", relevance="strong")
    result = score_fit(j)
    assert result.score == 100
    assert result.verdict == "strong"
    assert result.gate_failed is False
    assert result.judgment is j  # the evidence is carried through


def test_partial_coverage_arithmetic() -> None:
    # required = [met, unmet] -> coverage 0.5; no preferred; seniority adjacent
    # (0.6); domain partial (0.6).
    # weighted = 0.50*0.5 + 0.25*0.6 + 0.10*0.6 = 0.46 over total weight 0.85
    # -> round(100 * 0.46 / 0.85) = 54
    j = _judgment(
        [_req("required", "met"), _req("required", "unmet")],
        alignment="adjacent",
        relevance="partial",
    )
    result = score_fit(j)
    assert result.score == 54
    assert result.verdict == "weak"


def test_failed_knockout_caps_score_and_forces_none() -> None:
    # An unmet gate requirement must cap the score and force verdict "none",
    # regardless of strong seniority/domain.
    j = _judgment([_req("required", "unmet", is_gate=True)], alignment="exact", relevance="strong")
    result = score_fit(j)
    assert result.gate_failed is True
    assert result.verdict == "none"
    assert result.score is not None and result.score <= 20


def test_absent_preferred_dimension_is_renormalized_not_penalized() -> None:
    # An unmet preferred requirement drags the score down...
    with_unmet_preferred = score_fit(
        _judgment([_req("required", "met"), _req("preferred", "unmet")])
    )
    # ...but having no preferred requirements at all must NOT — the dimension is
    # dropped and weights renormalized.
    without_preferred = score_fit(_judgment([_req("required", "met")]))

    assert with_unmet_preferred.score == 85
    assert without_preferred.score == 100
    assert without_preferred.score > with_unmet_preferred.score


def test_score_is_deterministic() -> None:
    j = _judgment([_req("required", "partial"), _req("preferred", "met")], alignment="adjacent")
    assert score_fit(j).score == score_fit(j).score


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, "strong"),
        (80, "strong"),
        (79, "moderate"),
        (60, "moderate"),
        (59, "weak"),
        (40, "weak"),
        (39, "none"),
        (0, "none"),
    ],
)
def test_verdict_bands(score: int, expected: str) -> None:
    assert _verdict(score) == expected
