"""Shared loading/ID/date logic for the synthetic-persona eval (B1) and the
CV-parsing eval (B3) — used by inject_synthetic.py, run_synthetic.py, and
eval_profile_parsing.py, all of which need the same three primitives: find the
personas, generate stable IDs for them, and compute time-invariant durations.

See eval/personas/README.md for the fixture design (why CVs are written before
ground truth, why `years` is only fixed for closed roles, etc.).
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

_PERSONAS_DIR = Path(__file__).parent / "personas"

# Namespaces synthetic profile/job UUIDs — deterministic and stable across runs
# (re-running inject_synthetic.py must produce the same IDs, not new rows every
# time), and namespaced so a synthetic ID can never collide with a real one.
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "synthetic.job-radar.local")

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


@dataclass(frozen=True)
class Persona:
    persona_id: str
    cv_text: str
    ground_truth: dict
    synthetic_jobs: list[dict]


def persona_ids() -> list[str]:
    """All persona IDs, discovered from the committed CV fixtures."""
    return sorted(p.stem.removesuffix(".cv") for p in _PERSONAS_DIR.glob("*.cv.txt"))


def load_persona(persona_id: str) -> Persona:
    cv_text = (_PERSONAS_DIR / f"{persona_id}.cv.txt").read_text()
    ground_truth = json.loads((_PERSONAS_DIR / f"{persona_id}.ground_truth.json").read_text())
    jobs_path = _PERSONAS_DIR / f"{persona_id}.synthetic_jobs.json"
    synthetic_jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    return Persona(persona_id, cv_text, ground_truth, synthetic_jobs)


def profile_uuid(persona_id: str) -> uuid.UUID:
    return uuid.uuid5(_UUID_NAMESPACE, f"profile:{persona_id}")


def job_uuid(persona_id: str, index: int) -> uuid.UUID:
    return uuid.uuid5(_UUID_NAMESPACE, f"job:{persona_id}:{index}")


def parse_mon_year(text: str) -> date:
    """Parse a ground-truth 'Mon YYYY' string (e.g. 'Jan 2023') into a date.

    Ground-truth dates are always written in this fixed format (see the five
    eval/personas/*.ground_truth.json files) — this is not meant to parse the
    model's own, inconsistently-formatted output; see eval_profile_parsing.py
    for how that's handled instead.
    """
    mon, year = text.split()
    return date(int(year), _MONTHS[mon], 1)


def years_between(start: date, end: date) -> float:
    """Whole-month-precision duration in years, rounded to 2 decimals.

    Matches how every `years` value in the ground-truth fixtures was computed
    by hand (verified against real month arithmetic before being committed) —
    reusing this function is what keeps the eval script and the fixtures
    talking about the same number.
    """
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return round(months / 12, 2)


def today() -> date:
    return datetime.now(UTC).date()
