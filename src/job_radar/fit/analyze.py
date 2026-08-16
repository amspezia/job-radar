import logging

from job_radar.adapters.generation import generate
from job_radar.config import settings
from job_radar.db.models import Job, Profile
from job_radar.fit.schema import FitAssessment, FitJudgment
from job_radar.fit.score import score_fit

logger = logging.getLogger(__name__)

# Identifies the (prompt, FitJudgment schema) pair that produced an assessment.
# Bump on ANY change to _PROMPT or the schema: cached assessments are keyed on it,
# and serving a row generated under different instructions is a correctness bug,
# not a stale-cache annoyance.
PROMPT_VERSION = 2

_INSUFFICIENT_INPUT = FitAssessment(
    score=None, verdict="none", gate_failed=False, judgment=None, summary="insufficient input"
)
_GENERATION_FAILED = FitAssessment(
    score=None, verdict="none", gate_failed=False, judgment=None, summary="fit analysis failed"
)

_PROMPT = """\
You are assessing how well a candidate fits a job posting. Judge ONLY from the
facts given below — do not assume anything not stated.

For every requirement you identify in the posting, in this order:
- FIRST, gather evidence: a verbatim quote from the posting (source "posting")
  stating the requirement — quote it exactly rather than restating it in your own
  words — and, where the candidate's profile or CV speaks to it, a verbatim quote
  (source "profile") showing what they have done
- THEN classify it as "required" or "preferred"
- THEN classify satisfaction as "met", "partial", or "unmet", judged against the
  profile quote you just gathered

IMPORTANT — judge each requirement on its OWN merits against the candidate's
skills and experience. The company's industry/domain must NOT influence how you
rate individual technical or role requirements. A Python requirement is equally
"met" whether the posting is from a fintech company or a healthcare company.

Do NOT judge the candidate's location, work authorization, or region eligibility
— those are checked separately and deterministically.

Also judge:
- domain: how relevant the candidate's DOMAIN EXPERIENCE (not tech skills) is
  to the posting's business domain. This is ONLY about industry overlap
  (e.g. fintech↔fintech = strong, fintech↔e-commerce = partial,
  fintech↔biotech = weak). Tech stack match does NOT factor into domain.

Do NOT judge seniority/level — it is handled separately from posting metadata.

Do not include a numeric score anywhere — it is computed separately.

Keep the summary to a single sentence.

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
{posting_body}
"""


def _posting_body(posting: Job) -> str:
    """Build the posting section of the fit prompt from the richest available content.

    Structured fields (pre-extracted at ingest) are preferred: they exclude company
    boilerplate, benefits, and culture copy that the LLM would otherwise have to
    filter out manually, and they fit comfortably within num_ctx. Raw description is
    used only when both extracted fields are absent (extraction failed or skipped).
    """
    if posting.requirements or posting.responsibilities:
        parts: list[str] = []
        if posting.requirements:
            parts.append(f"requirements:\n{posting.requirements}")
        if posting.responsibilities:
            parts.append(f"responsibilities:\n{posting.responsibilities}")
        return "\n\n".join(parts)
    return f"description:\n{posting.description}"


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
        posting_body=_posting_body(posting),
    )


def _has_sufficient_input(profile: Profile, posting: Job) -> bool:
    keywords = profile.domains_keywords or {}
    has_profile_signal = bool(keywords.get("tech_stack")) or bool(profile.work_history)
    has_posting_signal = bool(posting.description and posting.description.strip())
    return has_profile_signal and has_posting_signal


async def analyze_fit(
    profile: Profile, posting: Job, *, levels: list[str] | None = None, model: str | None = None
) -> FitAssessment:
    """Judge a profile against a posting with the local LLM, then score it.

    The LLM only ever produces grounded classifications (FitJudgment); the
    numeric score is computed deterministically by score_fit. `levels` overrides
    the profile's accepted seniority levels for this call; `model` overrides the
    generation model (defaults to the configured fit/generation model).
    """
    if not _has_sufficient_input(profile, posting):
        logger.info("Skipping fit analysis for job %s: insufficient input", posting.id)
        return _INSUFFICIENT_INPUT

    logger.info("Analyzing fit for job %s", posting.id)
    prompt = _build_prompt(profile, posting)
    try:
        judgment = await generate(prompt, FitJudgment, model=model or settings.fit_model)
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
