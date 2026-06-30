from job_radar.db.models import Profile
from job_radar.fit.pipeline import build_query
from job_radar.retrieval.filters import build_profile_filter


def _sql(clause) -> str:
    return str(clause.compile(compile_kwargs={"literal_binds": True}))


def _profile(**over) -> Profile:
    base = {
        "location_rules": {},
        "remote_required": False,
        "salary_floor": None,
        "currency": None,
        "target_titles": [],
        "domains_keywords": {},
        "seniority": None,
        "seniority_rules": {},
    }
    base.update(over)
    return Profile(**base)


def test_no_constraints_returns_none() -> None:
    assert build_profile_filter(_profile()) is None


def test_geo_keywords_produce_location_clause() -> None:
    clause = build_profile_filter(_profile(location_rules={"allowed_keywords": ["brazil"]}))
    assert clause is not None and "location" in _sql(clause).lower()


def test_remote_required_filters_on_remote() -> None:
    clause = build_profile_filter(_profile(remote_required=True))
    assert clause is not None and "remote" in _sql(clause).lower()


def test_salary_floor_keeps_null_salary_rows() -> None:
    sql = _sql(build_profile_filter(_profile(salary_floor=120000, currency="USD"))).lower()
    # Excludes only confidently-too-low rows: NULL salary and other currencies survive.
    assert "salary_max" in sql
    assert "is null" in sql
    assert "currency" in sql


def test_seniority_filters_on_allowed_levels_keeping_null() -> None:
    # Senior candidate: keep NULL (unknown) or any in-range level; Principal
    # (out of range by default) is absent from the IN-list.
    sql = _sql(build_profile_filter(_profile(seniority="senior"))).lower()
    assert "seniority is null" in sql
    assert "seniority in" in sql
    assert "'senior'" in sql and "'staff'" in sql  # in range
    assert "'principal'" not in sql  # excluded


def test_explicit_allowed_levels_narrow_the_in_list() -> None:
    profile = _profile(seniority="senior", seniority_rules={"allowed_levels": ["senior"]})
    sql = _sql(build_profile_filter(profile)).lower()
    assert "'senior'" in sql
    assert "'staff'" not in sql  # now excluded


def test_unknown_candidate_level_adds_no_seniority_clause() -> None:
    # Accepts every level -> nothing to filter.
    assert build_profile_filter(_profile(seniority=None)) is None


def test_levels_override_narrows_what_the_profile_would_allow() -> None:
    # Default for a Senior allows Staff; the per-run override drops it.
    profile = _profile(seniority="senior")
    assert "'staff'" in _sql(build_profile_filter(profile)).lower()
    assert "'staff'" not in _sql(build_profile_filter(profile, levels=["senior"])).lower()


def test_build_query_merges_titles_stack_and_domains() -> None:
    profile = _profile(
        target_titles=["Backend Engineer"],
        domains_keywords={"tech_stack": ["Python", "Kafka"], "domains": ["fintech"]},
    )
    query = build_query(profile)
    assert query == "Backend Engineer Python Kafka fintech"


def test_build_query_handles_empty_profile() -> None:
    assert build_query(_profile()) == ""
