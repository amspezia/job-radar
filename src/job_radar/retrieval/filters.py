from sqlalchemy import and_, or_
from sqlalchemy.sql.elements import ColumnElement

from job_radar.db.models import Job, Profile
from job_radar.retrieval.geo import build_geo_filter
from job_radar.retrieval.seniority import LADDER, allowed_levels


def build_profile_filter(
    profile: Profile, *, levels: list[str] | None = None
) -> ColumnElement[bool] | None:
    """Combine the profile's hard preferences into one retrieval WHERE clause.

    Region (geo), remote, and salary floor are non-negotiable filters applied
    before ranking. `levels` overrides the profile's accepted seniority levels
    for this call (e.g. from a CLI flag). Returns None when the profile
    expresses no constraints.
    """
    clauses: list[ColumnElement[bool]] = []

    keywords = (profile.location_rules or {}).get("allowed_keywords")
    if keywords:
        clauses.append(build_geo_filter(keywords))

    if profile.remote_required:
        clauses.append(Job.remote.is_(True))

    # Exclude postings whose level is known and outside the accepted range.
    # NULL seniority is "unknown" and always kept (never exclude on missing data).
    # Intersect with LADDER to drop any invalid values that may appear in an
    # explicit seniority_rules config — unknown strings would make the set not a
    # strict subset of LADDER, silently disabling the filter.
    allowed = list({lvl for lvl in (levels or allowed_levels(profile)) if lvl in LADDER})
    if set(allowed) < set(LADDER):
        clauses.append(or_(Job.seniority.is_(None), Job.seniority.in_(allowed)))

    # Never drop a posting just because it omits salary (most do). Only exclude
    # one we can confidently judge too low: a known upper bound, in the
    # candidate's own currency, that still falls below the floor. Without a
    # declared currency we cannot compare numbers, so skip the filter entirely.
    if profile.salary_floor is not None and profile.currency is not None:
        clauses.append(
            or_(
                Job.salary_max.is_(None),
                Job.salary_max >= profile.salary_floor,
                Job.currency.is_distinct_from(profile.currency),
            )
        )

    if not clauses:
        return None
    return and_(*clauses)
