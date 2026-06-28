import hashlib
import re
from typing import NamedTuple

from bs4 import BeautifulSoup

from job_radar.ingest.base import NormalizedJob

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}

# A number with optional thousands commas and an optional k/K suffix:
# "30k", "50,000", "100K" all match.
_NUMBER_RE = re.compile(r"(\d[\d,]*)\s*([kK])?")


class ParsedSalary(NamedTuple):
    min: int | None
    max: int | None
    currency: str | None


def html_to_text(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)


def _to_int(digits: str, k_suffix: str) -> int:
    value = int(digits.replace(",", ""))
    if k_suffix:
        value *= 1000
    return value


def parse_salary(raw: str) -> ParsedSalary:
    """Best-effort parse of a free-text salary string into (min, max, currency).

    Handles the common forms ("$30k - $100k", "$50,000 - $80,000", "$100k").
    Anything unrecognised (empty, "Competitive", no numbers) yields all-None,
    honouring the "store unknown, never fabricate" rule.
    """
    if not raw:
        return ParsedSalary(None, None, None)

    currency = next((code for sym, code in _CURRENCY_SYMBOLS.items() if sym in raw), None)
    numbers = [_to_int(digits, k) for digits, k in _NUMBER_RE.findall(raw)]

    if not numbers:
        return ParsedSalary(None, None, currency)
    if len(numbers) == 1:
        return ParsedSalary(numbers[0], None, currency)
    return ParsedSalary(numbers[0], numbers[1], currency)


def content_hash(job: NormalizedJob) -> str:
    location = "none" if not job.location else job.location
    normalized_fields = [" ".join(s.split()).lower() for s in [job.company, job.title, location]]
    return hashlib.sha256("|".join(normalized_fields).encode("utf-8")).hexdigest()
