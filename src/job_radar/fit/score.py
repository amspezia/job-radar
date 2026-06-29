from job_radar.fit.schema import FitAssessment, FitJudgment, Requirement

# Per-judgment numeric values. Partial credit keeps the score smooth rather than
# all-or-nothing.
_SATISFACTION = {"met": 1.0, "partial": 0.5, "unmet": 0.0}
_SENIORITY = {"exact": 1.0, "adjacent": 0.6, "mismatch": 0.2}
_DOMAIN = {"strong": 1.0, "partial": 0.6, "weak": 0.2}

# Dimension weights (tunable; calibrated against human labels in M6). A dimension
# with no items is dropped and the remaining weights are renormalized, so a
# posting isn't penalized for, say, listing no "preferred" requirements.
_WEIGHTS = {"required": 0.50, "preferred": 0.15, "seniority": 0.25, "domain": 0.10}

# A failed knockout is non-compensatory: strengths elsewhere cannot offset it, so
# the score is capped low and the verdict forced to "none".
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


def score_fit(judgment: FitJudgment) -> FitAssessment:
    """Compute a 0-100 fit score and verdict from grounded judgments.

    Deterministic: identical judgments always yield the identical score. The LLM
    supplies the classifications; all arithmetic happens here.
    """
    required = [r for r in judgment.requirements if r.kind == "required"]
    preferred = [r for r in judgment.requirements if r.kind == "preferred"]

    # (weight key, 0-1 subscore) for each dimension that has something to score.
    dimensions: list[tuple[str, float]] = []
    if required:
        dimensions.append(("required", _coverage(required)))
    if preferred:
        dimensions.append(("preferred", _coverage(preferred)))
    dimensions.append(("seniority", _SENIORITY[judgment.seniority.alignment]))
    dimensions.append(("domain", _DOMAIN[judgment.domain.relevance]))

    total_weight = sum(_WEIGHTS[key] for key, _ in dimensions)
    weighted = sum(_WEIGHTS[key] * subscore for key, subscore in dimensions)
    score = round(100 * weighted / total_weight)

    gate_failed = any(r.is_gate and r.satisfaction == "unmet" for r in judgment.requirements)
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
