import logging
import re
from datetime import UTC, date, datetime

from job_radar.adapters.generation import generate
from job_radar.profile.schema import StructuredProfile

logger = logging.getLogger(__name__)

_PROMPT = """
You are extracting structured facts from a candidate's CV.
Extract ONLY information explicitly present in the CV text below.
Do NOT invent skills, roles, employers, or experience that are not stated.
If a field is not present, use null (or an empty list).

Today's date is {today} — use it if you need to judge how long an ongoing
("Present"/"Current") role has lasted.

For each work_history entry, capture role, company, start, end (exactly as
written on the CV, e.g. "May 2020" or "Present"), and the notable
achievements/responsibilities as `highlights`. `highlights` must be verbatim
quotes copied directly from the CV text — do not paraphrase, summarize, or
combine multiple bullets into one.

For `seniority`, return exactly one of: intern / junior / mid / senior / staff /
principal — inferred from years of experience and the most recent job title.
This list is ordered from least to most senior (so, for example, senior to
staff is one level and senior to principal is two). Return null if you
cannot confidently determine the level.

For `target_titles`, return 3-6 job titles the candidate should search for in
their NEXT role, consistent with the `seniority` level above (or, if you
returned `seniority` as null, consistent with their current, most recent job
title as stated) and their most recent/primary role.
- Do NOT simply copy titles that appear in `work_history` — in particular,
  exclude early-career titles (e.g. "Intern", "Junior ...") that no longer
  match the candidate's current level, unless their current, most recent
  title itself is at that level.
- Do NOT suggest a title more than one level above the candidate's current
  level unless the CV states scope explicitly matching that higher level
  (e.g. leading multiple teams, setting org-wide technical direction) —
  owning individual services or mentoring individual contributors is not
  sufficient evidence for a jump of more than one level.
- Do NOT suggest people-management titles ("Engineering Manager", "Lead",
  "Head of ...") unless the CV states actual people-management
  responsibility.

For `tech_stack`, list every specific skill, technology, language, framework,
or tool named anywhere in the CV — check the entire text, including
experience bullets and project descriptions, not only an explicit "Skills"
section if one exists. Use the term as written in the CV; do not infer a
technology from a description of what it does unless it is also named
explicitly. This field is extraction only — unlike `domains` below, no
inference is allowed here.

For `domains`, name the business/problem areas the candidate has worked in —
the industry or problem space their employer(s) operate in (e.g. "fintech",
"healthcare", "telecom"), not the technique or function the candidate
personally performed (e.g. not "risk prediction" or "data analysis" — those
describe *how*, not *what industry*). Weight this toward the candidate's
current/most recent role — an older employer's industry is only worth
including if it's still a meaningful part of their background, not simply
because it's mentioned. These may be inferred from the employers and project
descriptions even when not stated as a single word — this is the one field
where reasonable inference is allowed. Only return an empty or very short
list if the CV genuinely does not describe what industry or problem space any
employer operates in; if even one employer's business is described, that
description is a valid domain.

CV:
{cv_text}
"""

# The model is asked to preserve each work_history entry's start/end exactly as
# written on the CV, but in practice reliably ignores that and emits ISO-ish
# strings instead ('2023-01-01', '2019-01') alongside genuine 'Mon YYYY' text.
# `years`/`years_experience` are therefore never taken from the model's own
# arithmetic — multi-role date math has proven unreliable — they're computed
# here from whichever of these formats the extracted start/end actually is.
_MONTH_ABBR = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_ONGOING_MARKERS = {"", "present", "current", "now"}
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-\d{2})?$")
_MONTH_YEAR_RE = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{4})$")


def _is_ongoing(end: str | None) -> bool:
    return end is None or end.strip().lower() in _ONGOING_MARKERS


def _parse_cv_date(text: str | None) -> date | None:
    if not text:
        return None
    text = text.strip()
    if m := _ISO_DATE_RE.match(text):
        return date(int(m.group(1)), int(m.group(2)), 1)
    if m := _MONTH_YEAR_RE.match(text):
        month = _MONTH_ABBR.get(m.group(1)[:3].lower())
        if month:
            return date(int(m.group(2)), month, 1)
    return None


def _years_between(start: date, end: date) -> float:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return round(months / 12, 2)


def _with_computed_years(profile: StructuredProfile, today: date) -> StructuredProfile:
    """Overwrite every work_history `years` and top-level `years_experience`
    with values computed from the extracted start/end dates, rather than the
    model's own arithmetic. An entry whose start (or non-ongoing end) can't be
    parsed is left with years=None — an honest "unknown" beats a guess neither
    we nor the caller can distinguish from a computed value.
    """
    starts: list[date] = []
    new_items = []
    for item in profile.work_history:
        start = _parse_cv_date(item.start)
        end = today if _is_ongoing(item.end) else _parse_cv_date(item.end)
        years = _years_between(start, end) if start and end else None
        if start:
            starts.append(start)
        new_items.append(item.model_copy(update={"years": years}))

    years_experience = _years_between(min(starts), today) if starts else None
    return profile.model_copy(
        update={"work_history": new_items, "years_experience": years_experience}
    )


async def parse_cv(cv_text: str) -> StructuredProfile:
    # Log only the size, never the CV content (PII).
    logger.debug("Parsing CV (%d chars) into a structured profile", len(cv_text))
    today = datetime.now(UTC).date()
    prompt = _PROMPT.format(today=today.isoformat(), cv_text=cv_text)
    profile = await generate(prompt, StructuredProfile)
    return _with_computed_years(profile, today)
