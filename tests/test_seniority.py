from job_radar.db.models import Profile
from job_radar.retrieval.seniority import (
    allowed_levels,
    default_allowed,
    normalize_level,
)


def test_normalize_level_maps_synonyms() -> None:
    assert normalize_level("Senior") == "senior"
    assert normalize_level("Sr. Backend Engineer") == "senior"
    assert normalize_level("Staff Software Engineer") == "staff"
    assert normalize_level("Engenheiro Pleno") == "mid"


def test_normalize_level_unknown_and_compound() -> None:
    assert normalize_level("Backend Engineer") is None  # no level stated -> unknown
    assert normalize_level(None) is None
    assert normalize_level("Senior Staff Engineer") == "staff"  # resolves to the higher


def test_default_allowed_is_at_or_below_plus_one() -> None:
    allowed = default_allowed("senior")
    assert "staff" in allowed  # one above is reachable
    assert "principal" not in allowed  # two above is not
    assert "junior" in allowed  # below is fine


def test_default_allowed_unknown_candidate_accepts_all() -> None:
    assert default_allowed(None) == default_allowed("not-a-level")


def test_allowed_levels_prefers_explicit_rule() -> None:
    profile = Profile(seniority="senior", seniority_rules={"allowed_levels": ["mid", "senior"]})
    assert allowed_levels(profile) == ["mid", "senior"]
