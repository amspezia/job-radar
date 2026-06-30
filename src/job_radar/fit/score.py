from job_radar.db.models import Job, Profile
from job_radar.fit.schema import FitAssessment, FitJudgment, Requirement
from job_radar.retrieval.geo import region_allowed
from job_radar.retrieval.seniority import allowed_levels, normalize_level, rank

# Per-judgment numeric values. Partial credit keeps the score smooth rather than
# all-or-nothing.
_SATISFACTION = {"met": 1.0, "partial": 0.5, "unmet": 0.0}
_DOMAIN = {"strong": 1.0, "partial": 0.6, "weak": 0.2}

# Seniority subscore for a posting within the accepted range, by signed distance
# from the candidate's level (posting_rank - candidate_rank). Asymmetric: being
# under-qualified (posting above you) bites harder than being over-qualified.
_SENIORITY_UNKNOWN = 0.7  # posting states no level — neutral, never gated


def _seniority_subscore(delta: int) -> float:
    if delta == 0:
        return 1.0
    if delta < 0:  # over-qualified: gentle, floored
        return max(0.55, 1.0 + 0.15 * delta)
    return 0.25 if delta == 1 else 0.1  # under-qualified


# Dimension weights (tunable; calibrated against human labels in M6). A dimension
# with no items is dropped and the remaining weights are renormalized, so a
# posting isn't penalized for, say, listing no "preferred" requirements.
_WEIGHTS = {"required": 0.50, "preferred": 0.15, "seniority": 0.25, "domain": 0.10}

# A failed knockout is non-compensatory: strengths elsewhere cannot offset it, so
# the score is capped low and the verdict forced to "none". The only knockout is
# region eligibility, which is determined deterministically (never by the LLM).
_GATE_CAP = 20

# Score -> verdict bands.
_BANDS = ((80, "strong"), (60, "moderate"), (40, "weak"))


def _coverage(requirements: list[Requirement]) -> float:
    return sum(_SATISFACTION[r.satisfaction] for r in requirements) / len(requirements)


def _verdict(score: int) -> str:
    for threshold, label in _BANDS:
        if score >= threshold:
            return label
    return "none"


def score_fit(
    judgment: FitJudgment, job: Job, profile: Profile, *, levels: list[str] | None = None
) -> FitAssessment:
    """Compute a 0-100 fit score and verdict from grounded judgments.

    Deterministic: identical inputs always yield the identical score. The LLM
    supplies the requirement/seniority/domain classifications; all arithmetic —
    and the region/seniority knockouts — happen here. `levels` overrides the
    profile's accepted seniority levels for this call.
    """
    keywords = (profile.location_rules or {}).get("allowed_keywords") or []
    region_ok = region_allowed(job, keywords)

    # Seniority is structured metadata on the posting (set at ingest). A known
    # level outside the accepted range is a knockout; NULL is "unknown" — neutral
    # and never gated; in-range scores by distance from the candidate.
    posting_level = job.seniority
    allowed = levels or allowed_levels(profile)
    candidate_rank = rank(normalize_level(profile.seniority))
    if posting_level is None:
        seniority_ok, seniority_subscore = True, _SENIORITY_UNKNOWN
    elif posting_level not in allowed:
        seniority_ok, seniority_subscore = False, 0.0
    elif candidate_rank is None:
        seniority_ok, seniority_subscore = True, _SENIORITY_UNKNOWN
    else:
        seniority_ok = True
        seniority_subscore = _seniority_subscore(rank(posting_level) - candidate_rank)

    # Domain only scores against a real candidate signal — with no declared
    # domains the LLM's relevance call is ungrounded, so drop the dimension.
    domains = (profile.domains_keywords or {}).get("domains") or []
    score_domain = bool(domains)

    required = [r for r in judgment.requirements if r.kind == "required"]
    preferred = [r for r in judgment.requirements if r.kind == "preferred"]

    # (weight key, 0-1 subscore) for each dimension that has something to score.
    dimensions: list[tuple[str, float]] = []
    if required:
        dimensions.append(("required", _coverage(required)))
    if preferred:
        dimensions.append(("preferred", _coverage(preferred)))
    dimensions.append(("seniority", seniority_subscore))
    if score_domain:
        dimensions.append(("domain", _DOMAIN[judgment.domain.relevance]))

    total_weight = sum(_WEIGHTS[key] for key, _ in dimensions)
    weighted = sum(_WEIGHTS[key] * subscore for key, subscore in dimensions)
    score = round(100 * weighted / total_weight)

    gate_failed = not region_ok or not seniority_ok
    if gate_failed:
        score = min(score, _GATE_CAP)
        verdict = "none"
    else:
        verdict = _verdict(score)

    return FitAssessment(
        score=score,
        verdict=verdict,
        gate_failed=gate_failed,
        judgment=judgment,
        summary=judgment.summary,
    )
