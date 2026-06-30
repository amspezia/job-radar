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
    allowed = levels or allowed_levels(profile)
    if set(allowed) < set(LADDER):
        clauses.append(or_(Job.seniority.is_(None), Job.seniority.in_(allowed)))

    if profile.salary_floor is not None:
        # Never drop a posting just because it omits salary (most do). Only
        # exclude one we can confidently judge too low: a known upper bound,
        # in the candidate's own currency, that still falls below the floor.
        keep = or_(Job.salary_max.is_(None), Job.salary_max >= profile.salary_floor)
        if profile.currency is not None:
            keep = or_(keep, Job.currency.is_distinct_from(profile.currency))
        clauses.append(keep)

    if not clauses:
        return None
    return and_(*clauses)
