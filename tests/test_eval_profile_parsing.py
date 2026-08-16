from datetime import date

from eval.eval_profile_parsing import (
    check_acceptable_set,
    check_seniority,
    check_tech_stack,
    check_work_history_years,
    check_years_experience,
)


def test_seniority_exact_match_passes() -> None:
    assert check_seniority("senior", "senior").passed is True


def test_seniority_mismatch_fails() -> None:
    result = check_seniority(None, "junior")
    assert result.passed is False
    assert "junior" in result.detail


def test_acceptable_set_passes_with_expected_present() -> None:
    rule = {"expected_any_of": ["Senior Backend Engineer"], "must_not_include": ["Intern"]}
    result = check_acceptable_set(
        "target_titles", ["Senior Backend Engineer", "Staff Engineer"], rule
    )
    assert result.passed is True


def test_acceptable_set_loose_match_is_case_and_substring_tolerant() -> None:
    rule = {"expected_any_of": ["Machine Learning Engineer"], "must_not_include": []}
    result = check_acceptable_set("target_titles", ["machine learning engineer ii"], rule)
    assert result.passed is True


def test_acceptable_set_fails_when_nothing_expected_present() -> None:
    rule = {"expected_any_of": ["Senior Backend Engineer"], "must_not_include": []}
    result = check_acceptable_set("target_titles", ["Product Manager"], rule)
    assert result.passed is False


def test_acceptable_set_does_not_flag_substring_of_a_longer_forbidden_phrase() -> None:
    # Regression: "Backend Engineer" must not be flagged just because
    # "Principal Backend Engineer" is forbidden — that's the reverse
    # containment direction from a real violation.
    rule = {
        "expected_any_of": ["Backend Engineer"],
        "must_not_include": ["Principal Backend Engineer", "Engineering Manager"],
    }
    result = check_acceptable_set("target_titles", ["Backend Engineer"], rule)
    assert result.passed is True


def test_acceptable_set_still_catches_the_actual_forbidden_phrase() -> None:
    rule = {"expected_any_of": [], "must_not_include": ["Principal Backend Engineer"]}
    result = check_acceptable_set("target_titles", ["Principal Backend Engineer"], rule)
    assert result.passed is False


def test_acceptable_set_fails_on_forbidden_term_even_if_expected_present() -> None:
    rule = {"expected_any_of": ["Backend Engineer"], "must_not_include": ["Engineering Manager"]}
    result = check_acceptable_set(
        "target_titles", ["Backend Engineer", "Engineering Manager"], rule
    )
    assert result.passed is False
    assert "Engineering Manager" in result.detail


def test_acceptable_set_with_no_expected_constraint_only_checks_forbidden() -> None:
    # Dave's domains case: expected_any_of=[] means "no domain required".
    rule = {"expected_any_of": [], "must_not_include": []}
    result = check_acceptable_set("domains", [], rule)
    assert result.passed is True


def test_tech_stack_exact_match() -> None:
    result = check_tech_stack(["Rust", "PostgreSQL"], ["Rust", "PostgreSQL"])
    assert result.passed is True


def test_tech_stack_case_insensitive() -> None:
    result = check_tech_stack(["rust", "postgresql"], ["Rust", "PostgreSQL"])
    assert result.passed is True


def test_tech_stack_missing_item_fails() -> None:
    result = check_tech_stack(["Rust"], ["Rust", "PostgreSQL"])
    assert result.passed is False
    assert "postgresql" in result.detail


def test_tech_stack_invented_item_fails() -> None:
    result = check_tech_stack(["Rust", "PostgreSQL", "AWS"], ["Rust", "PostgreSQL"])
    assert result.passed is False
    assert "aws" in result.detail


def test_work_history_years_within_tolerance_passes() -> None:
    gt = [{"company": "Meridian Pay", "years": 2.5, "start": "Jun 2020", "end": "Dec 2022"}]
    parsed = [{"company": "Meridian Pay", "years": 2.6, "start": "Jun 2020", "end": "Dec 2022"}]
    checks = check_work_history_years(parsed, gt)
    assert len(checks) == 1
    assert checks[0].passed is True


def test_work_history_years_outside_tolerance_fails() -> None:
    gt = [{"company": "Meridian Pay", "years": 2.5, "start": "Jun 2020", "end": "Dec 2022"}]
    parsed = [{"company": "Meridian Pay", "years": 4.0, "start": "Jun 2020", "end": "Dec 2022"}]
    checks = check_work_history_years(parsed, gt)
    assert checks[0].passed is False


def test_work_history_years_skips_ongoing_roles() -> None:
    gt = [{"company": "Meridian Pay", "years": None, "start": "Jan 2023", "end": None}]
    checks = check_work_history_years([], gt)
    assert checks == []


def test_work_history_years_missing_company_fails() -> None:
    gt = [{"company": "Meridian Pay", "years": 2.5, "start": "Jun 2020", "end": "Dec 2022"}]
    checks = check_work_history_years(
        [{"company": "Some Other Co", "years": 2.5, "start": "Jun 2020", "end": "Dec 2022"}], gt
    )
    assert checks[0].passed is False
    assert "no parsed entry" in checks[0].detail


def test_work_history_years_ignores_ongoing_parsed_entry_for_closed_role() -> None:
    # Regression: a same-company ongoing role (end=None/"Present") must never
    # be matched against a closed ground-truth role's expected years, even
    # when it's the only candidate in the input.
    gt = [{"company": "Meridian Pay", "years": 2.5, "start": "Jun 2020", "end": "Dec 2022"}]
    parsed = [{"company": "Meridian Pay", "years": 3.5, "start": "Jan 2023", "end": "Present"}]
    checks = check_work_history_years(parsed, gt)
    assert checks[0].passed is False
    assert "no parsed entry" in checks[0].detail


def test_work_history_years_disambiguates_same_company_by_start_date() -> None:
    # The alice/dave regression: two closed roles at the same company must
    # each be paired with their own ground-truth entry, not both matched to
    # whichever parsed entry comes first.
    gt = [
        {"company": "Cartway Retail", "years": 3.17, "start": "Jan 2019", "end": "Mar 2022"},
        {"company": "Cartway Retail", "years": 3.33, "start": "Aug 2015", "end": "Dec 2018"},
    ]
    parsed = [
        {"company": "Cartway Retail", "years": 3.2, "start": "2019-01", "end": "2022-03"},
        {"company": "Cartway Retail", "years": 3.3, "start": "2015-08", "end": "2018-12"},
    ]
    checks = check_work_history_years(parsed, gt)
    assert checks[0].passed is True  # 3.2 vs 3.17
    assert checks[1].passed is True  # 3.3 vs 3.33


def test_years_experience_within_tolerance_passes() -> None:
    # career_start Aug 2018, "today" fixed at Aug 2026 -> expected 8.0 years.
    result = check_years_experience(8.2, "Aug 2018", date(2026, 8, 16))
    assert result.passed is True


def test_years_experience_outside_tolerance_fails() -> None:
    result = check_years_experience(6.5, "Aug 2018", date(2026, 8, 16))
    assert result.passed is False


def test_years_experience_null_fails() -> None:
    result = check_years_experience(None, "Aug 2018", date(2026, 8, 16))
    assert result.passed is False
    assert "null" in result.detail
