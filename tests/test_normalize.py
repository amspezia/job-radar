import pytest

from job_radar.ingest.base import NormalizedJob
from job_radar.ingest.normalize import content_hash, html_to_text, parse_salary


def _job(company="Acme", title="Senior Engineer", location="Worldwide") -> NormalizedJob:
    return NormalizedJob(
        source="remotive",
        source_type="aggregator",
        source_id="1",
        url="https://example.com/job/1",
        title=title,
        company=company,
        description="desc",
        salary_min=None,
        salary_max=None,
        currency=None,
        location=location,
        job_type="full_time",
        remote=True,
        published_at=None,
    )


class TestContentHash:
    def test_is_a_64_char_hex_digest(self) -> None:
        h = content_hash(_job())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_identical_jobs_hash_the_same(self) -> None:
        assert content_hash(_job()) == content_hash(_job())

    def test_whitespace_noise_does_not_change_the_hash(self) -> None:
        assert content_hash(_job()) == content_hash(_job(title="  Senior   Engineer "))

    def test_case_noise_does_not_change_the_hash(self) -> None:
        assert content_hash(_job()) == content_hash(_job(title="SENIOR ENGINEER"))

    def test_missing_location_matches_empty_string_location(self) -> None:
        assert content_hash(_job(location=None)) == content_hash(_job(location=""))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"title": "Junior Engineer"},
            {"company": "Globex"},
            {"location": "Remote - US only"},
        ],
    )
    def test_different_identity_fields_change_the_hash(self, overrides: dict) -> None:
        assert content_hash(_job()) != content_hash(_job(**overrides))


class TestParseSalary:
    @pytest.mark.parametrize(
        ("raw", "expected_min", "expected_max", "expected_currency"),
        [
            ("$30k - $100k", 30000, 100000, "USD"),
            ("$50,000 - $80,000", 50000, 80000, "USD"),
            ("$100k", 100000, None, "USD"),
            ("€40k-€60k", 40000, 60000, "EUR"),
            ("£90,000", 90000, None, "GBP"),
            ("Competitive", None, None, None),
            ("", None, None, None),
            ("50000", 50000, None, None),
        ],
    )
    def test_parses_common_forms(
        self,
        raw: str,
        expected_min: int | None,
        expected_max: int | None,
        expected_currency: str | None,
    ) -> None:
        result = parse_salary(raw)
        assert result.min == expected_min
        assert result.max == expected_max
        assert result.currency == expected_currency


class TestHtmlToText:
    def test_strips_tags(self) -> None:
        assert html_to_text("<p>Hello <strong>world</strong></p>") == "Hello world"

    def test_separates_list_items_with_spaces(self) -> None:
        assert html_to_text("<ul><li>Python</li><li>Postgres</li></ul>") == "Python Postgres"

    def test_unescapes_html_entities(self) -> None:
        assert html_to_text("Tom &amp; Jerry") == "Tom & Jerry"

    def test_empty_input_returns_empty_string(self) -> None:
        assert html_to_text("") == ""
