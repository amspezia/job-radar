"""The judge prompt: a static prefix, then the posting.

Everything candidate- and rubric-shaped comes first and never varies across the
~1.4k calls of a run, so Ollama reuses the cached prefix; the only per-call text
is the posting, sent as the user message.

The posting section is a byte-for-byte mirror of `eval/label.py::_show_job`
(minus its leading blank line and 2-space indent) — the view the user's blind labels
are graded on (F21, R-M3): title, company, source, seniority and the
*extracted* fields clipped to 400 chars, the description only when both
extracted fields are empty, and no location. Mirroring it is what makes a judge
grade comparable with a human one. It is deliberately duplicated rather than
imported: `eval.label` pulls in production modules the harness must not touch.
"""

import hashlib
import json
from collections.abc import Sequence

from eval.embedding.judge.base import CandidateBrief, JobView

# Bump when any part of the template below changes; recorded on every judge run.
PROMPT_VERSION = "3"

# Mirrors eval/label.py::_truncate's default — the label view's clip (F21).
LABEL_VIEW_CLIP = 400

# How much of the CV the brief carries. The rest is noise for a relevance call.
CV_CLIP = 3000

RUBRIC = """You are grading how relevant a job posting is to a specific candidate's job search.
Grades:
3 = Strong: exactly the role the candidate is searching for; right stack, level and domain.
2 = Relevant: the candidate would apply; maybe one thing is off but it belongs on the results page.
1 = Marginal: adjacent role, off-by-one seniority, or one key skill missing.
0 = Not relevant: wrong stack or domain, geo-blocked, or clearly off target.
How to apply the grades to this candidate (check in this order):
- Level: Staff, Lead, Principal, Head, Director, CTO and Manager roles are grade 0. A posting at \
the candidate's own level is fine, and so is one below it. If no level is stated, do not guess \
or penalize: judge from role and stack.
- Discipline: the core job must be software or AI engineering (backend, software engineer, AI \
engineer). Data engineering, data science or analytics, DevOps/SRE/platform-only, mobile, \
front-end-only and non-engineering jobs are grade 0. Machine-learning engineer and front-end \
heavy full-stack are grade 1 at most.
- Language and stack: Python is the best fit; Kotlin, JavaScript/TypeScript (Node) and SQL are the \
candidate's other languages. A backend or software-engineer role that accepts Python, is generic \
about the stack, or is built on JavaScript/TypeScript is grade 2 or 3. A backend role built around \
a language the candidate does not list (Go, Java-only, .NET/C#, Ruby-only, PHP) is grade 1. \
Generic skills (APIs, SQL, cloud, microservices, Kafka) are not overlap, but a missing framework \
alone is not a reason to downgrade a backend role.
- Grade 3: a backend, software or AI engineer role that accepts Python, or centers on the \
candidate's agent stack (MCP, agents, RAG, LLM evaluation). Grade 2: the same discipline with \
one thing off (JavaScript/TypeScript-centered, generic stack, or full-stack with a strong \
backend).
Judge only from the candidate brief and the posting shown. Give a one-sentence reason first, \
then the grade."""

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "grade": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": ["reason", "grade"],
}

JOB_HEADER = "# Posting to grade\n"

# The few-shot exemplars carry a human grade but no human rationale; a fixed
# per-grade line keeps the shape of the answer without inventing reasoning.
_EXAMPLE_REASONS = {
    3: "Squarely matches the role, stack and level.",
    2: "Would apply; belongs on the results page.",
    1: "Adjacent role or a key skill is missing.",
    0: "Wrong stack/domain or clearly off target.",
}


def _clip(text: str | None) -> str:
    if not text:
        return "(none)"
    text = text.strip()
    return text[:LABEL_VIEW_CLIP] + "…" if len(text) > LABEL_VIEW_CLIP else text


def render_label_view(job: JobView) -> str:
    """The posting exactly as a human labeler saw it."""
    lines = [
        f"Title   : {job.title}",
        f"Company : {job.company}",
        f"Source  : {job.source}  |  Seniority: {job.seniority or 'unknown'}",
    ]
    if job.requirements:
        lines.append(f"Requires: {_clip(job.requirements)}")
    if job.responsibilities:
        lines.append(f"Resp    : {_clip(job.responsibilities)}")
    if not job.requirements and not job.responsibilities:
        lines.append(f"Desc    : {_clip(job.description)}")
    return "\n".join(lines)


def _render_brief(brief: CandidateBrief) -> str:
    return (
        "# Candidate brief\n"
        f"seniority: {brief.seniority}\n"
        f"years_experience: {brief.years_experience}\n"
        f"target_titles: {', '.join(brief.target_titles)}\n"
        f"tech_stack: {', '.join(brief.tech_stack)}\n"
        f"domains: {', '.join(brief.domains)}\n"
        f"remote_required: {brief.remote_required}\n\n"
        f"# Candidate CV\n{brief.cv_text[:CV_CLIP]}"
    )


def build_static_prefix(brief: CandidateBrief, fewshot: Sequence[tuple[JobView, int]]) -> str:
    """Rubric, brief and graded examples — identical for every job of a run."""
    parts = [RUBRIC, _render_brief(brief)]
    if fewshot:
        examples = "\n\n".join(
            f"## Example {n}\n{render_label_view(job)}\n"
            f"=> {json.dumps({'reason': _EXAMPLE_REASONS[grade], 'grade': grade})}"
            for n, (job, grade) in enumerate(fewshot, start=1)
        )
        parts.append(f"# Graded examples\n{examples}")
    parts.append(JOB_HEADER)
    return "\n\n".join(parts)


def prompt_sha() -> str:
    """Hash of the candidate-independent template, stored on every judge run."""
    template = "\n".join(
        [PROMPT_VERSION, RUBRIC, json.dumps(JUDGMENT_SCHEMA, sort_keys=True), str(LABEL_VIEW_CLIP)]
    )
    return hashlib.sha256(template.encode("utf-8")).hexdigest()
