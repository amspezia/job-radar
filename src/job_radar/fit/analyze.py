import logging

from job_radar.adapters.generation import generate
from job_radar.db.models import Job, Profile
from job_radar.fit.schema import FitAssessment, FitJudgment
from job_radar.fit.score import score_fit

logger = logging.getLogger(__name__)

_INSUFFICIENT_INPUT = FitAssessment(
    score=None, verdict="none", gate_failed=False, judgment=None, summary="insufficient input"
)
_GENERATION_FAILED = FitAssessment(
    score=None, verdict="none", gate_failed=False, judgment=None, summary="fit analysis failed"
)

_PROMPT = """\
You are assessing how well a candidate fits a job posting. Judge ONLY from the
facts given below — do not assume anything not stated.

For every requirement you identify in the posting:
- classify it as "required" or "preferred"
- classify satisfaction as "met", "partial", or "unmet" based on the candidate's
  profile and CV
- attach at least one verbatim quote (source "profile" or "posting") backing
  the classification

Do NOT judge the candidate's location, work authorization, or region eligibility
— those are checked separately and deterministically.

Also judge:
- domain: how relevant the candidate's background is to the posting's domain
  ("strong", "partial", or "weak")

Do NOT judge seniority/level — it is handled separately from posting metadata.

Do not include a numeric score anywhere — it is computed separately.

# Candidate profile
seniority: {seniority}
target_titles: {target_titles}
tech_stack: {tech_stack}
domains: {domains}
work_history: {work_history}

# Candidate CV
{cv_text}

# Job posting
title: {title}
company: {company}
description:
{description}
"""


def _build_prompt(profile: Profile, posting: Job) -> str:
    keywords = profile.domains_keywords or {}
    return _PROMPT.format(
        seniority=profile.seniority,
        target_titles=", ".join(profile.target_titles or []),
        tech_stack=", ".join(keywords.get("tech_stack", [])),
        domains=", ".join(keywords.get("domains", [])),
        work_history=profile.work_history or [],
        cv_text=profile.cv_text or "(none provided)",
        title=posting.title,
        company=posting.company,
        description=posting.description,
    )


def _has_sufficient_input(profile: Profile, posting: Job) -> bool:
    keywords = profile.domains_keywords or {}
    has_profile_signal = bool(keywords.get("tech_stack")) or bool(profile.work_history)
    has_posting_signal = bool(posting.description and posting.description.strip())
    return has_profile_signal and has_posting_signal


async def analyze_fit(
    profile: Profile, posting: Job, *, levels: list[str] | None = None
) -> FitAssessment:
    """Judge a profile against a posting with the local LLM, then score it.

    The LLM only ever produces grounded classifications (FitJudgment); the
    numeric score is computed deterministically by score_fit. `levels` overrides
    the profile's accepted seniority levels for this call.
    """
    if not _has_sufficient_input(profile, posting):
        logger.info("Skipping fit analysis for job %s: insufficient input", posting.id)
        return _INSUFFICIENT_INPUT

    logger.info("Analyzing fit for job %s", posting.id)
    prompt = _build_prompt(profile, posting)
    try:
        judgment = await generate(prompt, FitJudgment)
    except Exception:
        # A malformed/truncated LLM response or a transient model error must not
        # abort the whole batch — degrade this one job and keep scoring the rest.
        logger.exception("Fit analysis failed for job %s", posting.id)
        return _GENERATION_FAILED
    assessment = score_fit(judgment, posting, profile, levels=levels)
    logger.info(
        "Fit analysis for job %s: score=%s verdict=%s gate_failed=%s",
        posting.id,
        assessment.score,
        assessment.verdict,
        assessment.gate_failed,
    )
    return assessment
