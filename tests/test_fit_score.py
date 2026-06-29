import pytest

from job_radar.db.models import Job, Profile
from job_radar.fit.schema import (
    DomainJudgment,
    FitJudgment,
    Requirement,
    SeniorityJudgment,
)
from job_radar.fit.score import _verdict, score_fit

# Region keywords mirroring the default profile rule set.
_KEYWORDS = ["brazil", "brasil", "br", "worldwide", "anywhere", "global"]


def _req(kind: str = "required", satisfaction: str = "met") -> Requirement:
    return Requirement(text="x", kind=kind, satisfaction=satisfaction, evidence=[])


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


def _job(
    *, location: str | None = "Brazil", title: str = "Backend Engineer", description: str = ""
) -> Job:
    return Job(location=location, title=title, description=description)


def _profile(
    *, keywords: list[str] | None = None, domains: list[str] | None = ("saas",)
) -> Profile:
    return Profile(
        location_rules={"allowed_keywords": keywords} if keywords is not None else {},
        domains_keywords={"domains": list(domains) if domains else []},
    )


def test_all_dimensions_maxed_scores_100_strong() -> None:
    j = _judgment([_req("required"), _req("preferred")], alignment="exact", relevance="strong")
    result = score_fit(j, _job(), _profile())
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
    result = score_fit(j, _job(), _profile())
    assert result.score == 54
    assert result.verdict == "weak"


def test_region_mismatch_caps_score_and_forces_none() -> None:
    # A posting whose location is outside the allowed regions is a non-compensatory
    # knockout, regardless of strong seniority/domain/coverage.
    j = _judgment([_req("required", "met")], alignment="exact", relevance="strong")
    job = _job(location="United Kingdom", description="a global collaborative culture")
    result = score_fit(j, job, _profile(keywords=_KEYWORDS))
    assert result.gate_failed is True
    assert result.verdict == "none"
    assert result.score is not None and result.score <= 20


def test_allowed_region_does_not_gate() -> None:
    j = _judgment([_req("required", "met")], alignment="exact", relevance="strong")
    job = _job(location="Brazil")
    result = score_fit(j, job, _profile(keywords=_KEYWORDS))
    assert result.gate_failed is False
    assert result.score == 100


def test_absent_preferred_dimension_is_renormalized_not_penalized() -> None:
    # An unmet preferred requirement drags the score down...
    with_unmet_preferred = score_fit(
        _judgment([_req("required", "met"), _req("preferred", "unmet")]), _job(), _profile()
    )
    # ...but having no preferred requirements at all must NOT — the dimension is
    # dropped and weights renormalized.
    without_preferred = score_fit(_judgment([_req("required", "met")]), _job(), _profile())

    assert with_unmet_preferred.score == 85
    assert without_preferred.score == 100
    assert without_preferred.score > with_unmet_preferred.score


def test_domain_dropped_when_profile_has_no_domains() -> None:
    # With no declared candidate domains, a weak domain relevance must not drag
    # the score: the dimension is dropped, not scored as 0.2.
    j = _judgment([_req("required", "met")], alignment="exact", relevance="weak")
    scored_with = score_fit(j, _job(), _profile(domains=["saas"]))
    scored_without = score_fit(j, _job(), _profile(domains=[]))
    assert scored_without.score == 100  # required + seniority only, both maxed
    assert scored_without.score > scored_with.score


def test_score_is_deterministic() -> None:
    j = _judgment([_req("required", "partial"), _req("preferred", "met")], alignment="adjacent")
    assert score_fit(j, _job(), _profile()).score == score_fit(j, _job(), _profile()).score


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
