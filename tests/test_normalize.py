import pytest

from job_radar.adapters.sources.normalize import html_to_text, parse_salary


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
            # European thousands-grouping with a period, not a comma.
            ("€68.000 — €91.000 EUR", 68000, 91000, "EUR"),
            # Decimal comma before a k suffix.
            ("$31,2k- $52k", 31200, 52000, "USD"),
            # Bare decimal with no k suffix and no period marker: too small
            # to be a real annual salary, dropped rather than fabricated.
            ("$55.12 — $68.89 USD", None, None, "USD"),
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
