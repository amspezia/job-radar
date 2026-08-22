import re

from sqlalchemy import func, or_
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job

# A location that says nothing beyond "this job is remote" (bare "Remote",
# "Fully Remote", "Distributed"...) carries no country/region signal at all --
# unlike an explicit "Remote - United States", it is not evidence the posting
# is ineligible for the candidate's region.
_NO_SIGNAL_LOCATION = (
    r"^(fully\s+|100%\s+)?remote(\s*-?\s*first)?$|^distributed$|^work(ing)?\s+from\s+home$"
)


def _pattern(keywords: list[str]) -> str:
    # Postgres's advanced regex uses \y for a word boundary — \b is a literal
    # backspace character there, not an assertion like in Perl/Python regex.
    return r"\y(" + "|".join(re.escape(k) for k in keywords) + r")\y"


def _py_pattern(keywords: list[str]) -> str:
    # Python's re uses \b for the same word-boundary assertion that Postgres
    # spells \y, so region_allowed() matches the SQL prefilter exactly.
    return r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"


def build_geo_filter(keywords: list[str]) -> ColumnElement[bool]:
    """Allow jobs whose location matches one of the given keywords, or whose
    location carries no country/region signal at all.

    The structured `location` field is authoritative when it names a place: if
    a posting states an explicit country/region, only that field is consulted
    (so "Senior Engineer | UK" is not rescued by the word "global" appearing
    elsewhere). But a blank location, or pure remote-work boilerplate like
    "Remote" with no place name, is not evidence of ineligibility -- excluding
    those would silently drop postings that may well be open to the
    candidate's region, so they pass through by default.
    """
    location_match = Job.location.op("~*")(_pattern(keywords))
    trimmed = func.btrim(Job.location)
    no_signal = or_(Job.location.is_(None), trimmed == "", trimmed.op("~*")(_NO_SIGNAL_LOCATION))
    return or_(location_match, no_signal)


def region_allowed(job: Job, keywords: list[str]) -> bool:
    """Python-side mirror of :func:`build_geo_filter` for a single posting.

    Used by fit scoring to gate region deterministically instead of asking the
    LLM. Same semantics as the retrieval prefilter.
    """
    if not keywords:
        return True

    location = (job.location or "").strip()
    if not location or re.fullmatch(_NO_SIGNAL_LOCATION, location, re.IGNORECASE):
        return True
    return re.search(_py_pattern(keywords), location, re.IGNORECASE) is not None
