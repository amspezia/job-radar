import logging

from job_radar.adapters.generation import generate
from job_radar.db.models import Job, Profile
from job_radar.fit.schema import FitAssessment, FitJudgment
from job_radar.fit.score import score_fit

logger = logging.getLogger(__name__)

_INSUFFICIENT_INPUT = FitAssessment(
    score=None, verdict="none", gate_failed=False, judgment=None, summary="insufficient input"
)

_PROMPT = """\
You are assessing how well a candidate fits a job posting. Judge ONLY from the
facts given below — do not assume anything not stated.

For every requirement you identify in the posting:
- classify it as "required" or "preferred"
- set is_gate=true only for genuine dealbreakers (work authorization, visa,
  language fluency, on-site/location mandates, security clearance) — NOT for
  ordinary skill requirements
- classify satisfaction as "met", "partial", or "unmet" based on the candidate's
  profile
- attach at least one verbatim quote (source "profile" or "posting") backing
  the classification

Also judge:
- seniority: the posting's required level, the candidate's level, and whether
  they are "exact", "adjacent", or "mismatch"
- domain: how relevant the candidate's background is to the posting's domain
  ("strong", "partial", or "weak")

Do not include a numeric score anywhere — it is computed separately.

# Candidate profile
seniority: {seniority}
target_titles: {target_titles}
tech_stack: {tech_stack}
domains: {domains}
work_history: {work_history}

# Job posting
title: {title}
company: {company}
location: {location}
remote: {remote}
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
        title=posting.title,
        company=posting.company,
        location=posting.location or "unspecified",
        remote=posting.remote,
        description=posting.description,
    )


def _has_sufficient_input(profile: Profile, posting: Job) -> bool:
    keywords = profile.domains_keywords or {}
    has_profile_signal = bool(keywords.get("tech_stack")) or bool(profile.work_history)
    has_posting_signal = bool(posting.description and posting.description.strip())
    return has_profile_signal and has_posting_signal


async def analyze_fit(profile: Profile, posting: Job) -> FitAssessment:
    """Judge a profile against a posting with the local LLM, then score it.

    The LLM only ever produces grounded classifications (FitJudgment); the
    numeric score is computed deterministically by score_fit.
    """
    if not _has_sufficient_input(profile, posting):
        logger.info("Skipping fit analysis for job %s: insufficient input", posting.id)
        return _INSUFFICIENT_INPUT

    logger.info("Analyzing fit for job %s", posting.id)
    prompt = _build_prompt(profile, posting)
    judgment = await generate(prompt, FitJudgment)
    assessment = score_fit(judgment)
    logger.info(
        "Fit analysis for job %s: score=%s verdict=%s gate_failed=%s",
        posting.id,
        assessment.score,
        assessment.verdict,
        assessment.gate_failed,
    )
    return assessment
