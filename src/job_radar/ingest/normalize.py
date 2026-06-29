import hashlib
import re
from typing import NamedTuple

from bs4 import BeautifulSoup

from job_radar.ingest.base import NormalizedJob

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}

# A figure has an integer part — either plain digits or thousands-grouped
# (comma or period, either convention: "50,000" or "68.000") — an optional
# decimal fraction (comma or period followed by 1-2 digits, e.g. "31,2" or
# "55.12"), and an optional k/K suffix: "30k", "50,000", "31,2k" all match.
_NUMBER_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d+)([.,]\d{1,2})?\s*([kK])?")

# A figure below this is never a real annual salary (whether it's a stray
# hourly rate like "$55.12/hr" with no period marker, or a decimal fraction
# without a k suffix). Filtering it out honours "store unknown, never
# fabricate" rather than reporting a clearly-wrong tiny number.
_MIN_PLAUSIBLE_SALARY = 1000


class ParsedSalary(NamedTuple):
    min: int | None
    max: int | None
    currency: str | None


def html_to_text(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)


def _to_value(integer_part: str, frac_part: str | None, k_suffix: str | None) -> int | None:
    base = int(re.sub(r"[.,]", "", integer_part))
    if frac_part:
        frac_digits = frac_part[1:]  # drop the leading separator
        value = base + int(frac_digits) / (10 ** len(frac_digits))
    else:
        value = float(base)
    if k_suffix:
        value *= 1000
    value = round(value)
    return value if value >= _MIN_PLAUSIBLE_SALARY else None


def parse_salary(raw: str) -> ParsedSalary:
    """Best-effort parse of a free-text salary string into (min, max, currency).

    Handles the common forms ("$30k - $100k", "$50,000 - $80,000", "$100k"),
    European thousands-grouping ("€68.000 - €91.000"), and decimal-before-k
    ("$31,2k"). Anything unrecognised (empty, "Competitive", no numbers, or a
    figure too small to be a real annual salary) yields all-None, honouring
    the "store unknown, never fabricate" rule.
    """
    if not raw:
        return ParsedSalary(None, None, None)

    currency = next((code for sym, code in _CURRENCY_SYMBOLS.items() if sym in raw), None)
    numbers = [_to_value(*m.groups()) for m in _NUMBER_RE.finditer(raw)]

    if not numbers:
        return ParsedSalary(None, None, currency)
    if len(numbers) == 1:
        return ParsedSalary(numbers[0], None, currency)
    return ParsedSalary(numbers[0], numbers[1], currency)


def content_hash(job: NormalizedJob) -> str:
    location = "none" if not job.location else job.location
    normalized_fields = [" ".join(s.split()).lower() for s in [job.company, job.title, location]]
    return hashlib.sha256("|".join(normalized_fields).encode("utf-8")).hexdigest()
