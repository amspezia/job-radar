import re

from sqlalchemy import and_, func, or_
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job

# Keywords this short (e.g. "br" for Brazil) collide with ordinary English
# words/abbreviations in free text ("BR cadence" = business review), so they
# only count as a match in the structured `location` field, never in
# title/description prose.
_LOCATION_ONLY_MAX_LEN = 2


def _pattern(keywords: list[str]) -> str:
    # Postgres's advanced regex uses \y for a word boundary — \b is a literal
    # backspace character there, not an assertion like in Perl/Python regex.
    return r"\y(" + "|".join(re.escape(k) for k in keywords) + r")\y"


def _py_pattern(keywords: list[str]) -> str:
    # Python's re uses \b for the same word-boundary assertion that Postgres
    # spells \y, so region_allowed() matches the SQL prefilter exactly.
    return r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"


def _prose_keywords(keywords: list[str]) -> list[str]:
    return [k for k in keywords if len(k) > _LOCATION_ONLY_MAX_LEN]


def build_geo_filter(keywords: list[str]) -> ColumnElement[bool]:
    """Allow only jobs whose hiring region matches one of the given keywords.

    The structured `location` field is authoritative: if a posting states a
    location, only that field is consulted (so "Senior Engineer | UK" is not
    rescued by the word "global" appearing in its marketing prose). Only when a
    posting has no location at all do we fall back to scanning the title and
    description for a country / "worldwide" / "anywhere" signal.
    """
    location_match = Job.location.op("~*")(_pattern(keywords))

    prose = _prose_keywords(keywords)
    prose_pattern = _pattern(prose)
    fallback = and_(
        or_(Job.location.is_(None), func.btrim(Job.location) == ""),
        or_(Job.title.op("~*")(prose_pattern), Job.description.op("~*")(prose_pattern)),
    )
    return or_(location_match, fallback)


def region_allowed(job: Job, keywords: list[str]) -> bool:
    """Python-side mirror of :func:`build_geo_filter` for a single posting.

    Used by fit scoring to gate region deterministically instead of asking the
    LLM. Same semantics as the retrieval prefilter: the structured location
    wins; title/description are consulted only when there is no location.
    """
    if not keywords:
        return True

    location = (job.location or "").strip()
    if location:
        return re.search(_py_pattern(keywords), location, re.IGNORECASE) is not None

    prose_pattern = _py_pattern(_prose_keywords(keywords))
    text = f"{job.title}\n{job.description or ''}"
    return re.search(prose_pattern, text, re.IGNORECASE) is not None
