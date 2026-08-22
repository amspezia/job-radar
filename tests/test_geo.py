from job_radar.db.models import Job
from job_radar.retrieval.geo import region_allowed

_KEYWORDS = ["brazil", "brasil", "br", "worldwide", "anywhere", "global"]


def _job(*, location=None, title="Backend Engineer", description="") -> Job:
    return Job(location=location, title=title, description=description)


def test_no_keywords_allows_everything() -> None:
    assert region_allowed(_job(location="United Kingdom"), []) is True


def test_matching_location_is_allowed() -> None:
    assert region_allowed(_job(location="Brazil"), _KEYWORDS) is True
    assert region_allowed(_job(location="Remote — Worldwide"), _KEYWORDS) is True


def test_location_present_but_non_matching_is_rejected_despite_prose() -> None:
    # "global"/"worldwide" in marketing prose must NOT rescue a posting that
    # states a non-allowed hiring location.
    job = _job(location="United Kingdom", description="an open, global collaborative culture")
    assert region_allowed(job, _KEYWORDS) is False


def test_null_location_is_allowed_by_default() -> None:
    assert region_allowed(_job(location=None, description="Join our team."), _KEYWORDS) is True


def test_bare_remote_location_is_allowed_by_default() -> None:
    # "Remote" alone is not evidence of being e.g. US-only -- unlike an
    # explicit "Remote - United States", it carries no country signal at all.
    assert region_allowed(_job(location="Remote"), _KEYWORDS) is True
    assert region_allowed(_job(location="Fully Remote"), _KEYWORDS) is True


def test_short_key_matches_structured_location_only() -> None:
    # "br" must match the structured location but never free text (would hit
    # "library", "abbreviation", ...). Free text is never consulted now, since
    # a missing/no-signal location already passes by default.
    assert region_allowed(_job(location="BR"), _KEYWORDS) is True


def test_word_boundary_prevents_substring_match() -> None:
    assert region_allowed(_job(location="Gibraltar"), _KEYWORDS) is False
