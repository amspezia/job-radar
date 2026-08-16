from datetime import date

from eval import personas_lib


def test_parse_mon_year() -> None:
    assert personas_lib.parse_mon_year("Jan 2023") == date(2023, 1, 1)
    assert personas_lib.parse_mon_year("Dec 2018") == date(2018, 12, 1)


def test_years_between_matches_hand_verified_fixture_values() -> None:
    # These pairs and expected values are exactly what's committed in the five
    # eval/personas/*.ground_truth.json files — re-verify the function that
    # produced them still produces them.
    cases = [
        (("Jun 2020", "Dec 2022"), 2.5),
        (("Aug 2018", "May 2020"), 1.75),
        (("Jul 2022", "Feb 2024"), 1.58),
        (("May 2021", "Jun 2022"), 1.08),
        (("Sep 2021", "May 2023"), 1.67),
        (("Jan 2019", "Mar 2022"), 3.17),
        (("Aug 2015", "Dec 2018"), 3.33),
        (("May 2024", "Dec 2024"), 0.58),
    ]
    for (start, end), expected in cases:
        result = personas_lib.years_between(
            personas_lib.parse_mon_year(start), personas_lib.parse_mon_year(end)
        )
        assert result == expected, f"{start}-{end}: expected {expected}, got {result}"


def test_years_between_same_month_is_zero() -> None:
    d = date(2024, 3, 1)
    assert personas_lib.years_between(d, d) == 0.0


def test_persona_ids_finds_all_five() -> None:
    ids = personas_lib.persona_ids()
    assert ids == [
        "alice-rust-senior",
        "bob-python-ml-mid",
        "carol-ts-fullstack-mid",
        "dave-go-platform-staff",
        "eve-data-junior",
    ]


def test_profile_uuid_is_deterministic() -> None:
    a = personas_lib.profile_uuid("alice-rust-senior")
    b = personas_lib.profile_uuid("alice-rust-senior")
    assert a == b


def test_profile_uuid_differs_per_persona() -> None:
    a = personas_lib.profile_uuid("alice-rust-senior")
    b = personas_lib.profile_uuid("bob-python-ml-mid")
    assert a != b


def test_job_uuid_is_deterministic_and_index_sensitive() -> None:
    a0 = personas_lib.job_uuid("alice-rust-senior", 0)
    a0_again = personas_lib.job_uuid("alice-rust-senior", 0)
    a1 = personas_lib.job_uuid("alice-rust-senior", 1)
    assert a0 == a0_again
    assert a0 != a1


def test_load_persona_returns_cv_ground_truth_and_jobs() -> None:
    persona = personas_lib.load_persona("alice-rust-senior")
    assert persona.persona_id == "alice-rust-senior"
    assert "Alice Renner" in persona.cv_text
    assert persona.ground_truth["seniority"] == "senior"
    assert len(persona.synthetic_jobs) == 20
