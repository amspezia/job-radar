import pytest

from job_radar.db.models import Job, Profile
from job_radar.fit.schema import DomainJudgment, FitJudgment, Requirement
from job_radar.fit.score import _verdict, score_fit

# Region keywords mirroring the default profile rule set.
_KEYWORDS = ["brazil", "brasil", "br", "worldwide", "anywhere", "global"]


def _req(kind: str = "required", satisfaction: str = "met") -> Requirement:
    return Requirement(text="x", kind=kind, satisfaction=satisfaction, evidence=[])


def _judgment(requirements: list[Requirement], relevance: str = "strong") -> FitJudgment:
    return FitJudgment(
        requirements=requirements,
        domain=DomainJudgment(relevance=relevance, evidence=[]),
        summary="ok",
    )


def _job(
    *,
    location: str | None = "Brazil",
    title: str = "Backend Engineer",
    description: str = "",
    seniority: str | None = "senior",
) -> Job:
    return Job(location=location, title=title, description=description, seniority=seniority)


def _profile(
    *,
    keywords: list[str] | None = None,
    domains: list[str] | None = ("saas",),
    seniority: str = "senior",
    seniority_rules: dict | None = None,
) -> Profile:
    return Profile(
        location_rules={"allowed_keywords": keywords} if keywords is not None else {},
        domains_keywords={"domains": list(domains) if domains else []},
        seniority=seniority,
        seniority_rules=seniority_rules if seniority_rules is not None else {},
    )


def test_all_dimensions_maxed_scores_100_strong() -> None:
    j = _judgment([_req("required"), _req("preferred")], relevance="strong")
    result = score_fit(j, _job(), _profile())
    assert result.score == 100
    assert result.verdict == "strong"
    assert result.gate_failed is False
    assert result.judgment is j  # the evidence is carried through


def test_partial_coverage_arithmetic() -> None:
    # required = [met, unmet] -> coverage 0.5; no preferred; seniority at-level
    # (1.0); domain partial (0.6).
    # weighted = 0.50*0.5 + 0.25*1.0 + 0.10*0.6 = 0.56 over total weight 0.85
    # -> round(100 * 0.56 / 0.85) = 66
    j = _judgment([_req("required", "met"), _req("required", "unmet")], relevance="partial")
    result = score_fit(j, _job(), _profile())
    assert result.score == 66
    assert result.verdict == "moderate"


def test_region_mismatch_caps_score_and_forces_none() -> None:
    # A posting whose location is outside the allowed regions is a non-compensatory
    # knockout, regardless of strong seniority/domain/coverage.
    j = _judgment([_req("required", "met")], relevance="strong")
    job = _job(location="United Kingdom", description="a global collaborative culture")
    result = score_fit(j, job, _profile(keywords=_KEYWORDS))
    assert result.gate_failed is True
    assert result.verdict == "none"
    assert result.score is not None and result.score <= 20


def test_allowed_region_does_not_gate() -> None:
    j = _judgment([_req("required", "met")], relevance="strong")
    job = _job(location="Brazil")
    result = score_fit(j, job, _profile(keywords=_KEYWORDS))
    assert result.gate_failed is False
    assert result.score == 100


def test_over_leveled_in_range_is_penalized_not_gated() -> None:
    # Staff is one above a Senior candidate: in range by default, but the
    # asymmetric penalty must pull it out of "strong".
    j = _judgment([_req("required", "met")], relevance="strong")
    result = score_fit(j, _job(seniority="staff"), _profile())
    assert result.gate_failed is False
    assert result.verdict != "strong"
    assert result.score is not None and result.score < 80


def test_far_over_leveled_is_gated() -> None:
    # Principal is 2+ above a Senior: outside the default range -> knockout.
    j = _judgment([_req("required", "met")], relevance="strong")
    result = score_fit(j, _job(seniority="principal"), _profile())
    assert result.gate_failed is True
    assert result.verdict == "none"
    assert result.score is not None and result.score <= 20


def test_unknown_level_is_neutral_and_never_gated() -> None:
    j = _judgment([_req("required", "met")], relevance="strong")
    result = score_fit(j, _job(seniority=None), _profile())
    assert result.gate_failed is False
    assert result.score is not None and result.score > 80  # neutral, not punished


def test_explicit_allowed_levels_can_gate_staff() -> None:
    # The user tightens the rule to exclude Staff entirely.
    j = _judgment([_req("required", "met")], relevance="strong")
    profile = _profile(seniority_rules={"allowed_levels": ["mid", "senior"]})
    result = score_fit(j, _job(seniority="staff"), profile)
    assert result.gate_failed is True
    assert result.verdict == "none"


def test_levels_override_gates_a_default_in_range_level() -> None:
    # Staff is in range by default for a Senior, but a per-run override excludes it.
    j = _judgment([_req("required", "met")], relevance="strong")
    job = _job(seniority="staff")
    assert score_fit(j, job, _profile()).gate_failed is False
    assert score_fit(j, job, _profile(), levels=["mid", "senior"]).gate_failed is True


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
    j = _judgment([_req("required", "met")], relevance="weak")
    scored_with = score_fit(j, _job(), _profile(domains=["saas"]))
    scored_without = score_fit(j, _job(), _profile(domains=[]))
    assert scored_without.score == 100  # required + seniority only, both maxed
    assert scored_without.score > scored_with.score


def test_score_is_deterministic() -> None:
    j = _judgment([_req("required", "partial"), _req("preferred", "met")])
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
