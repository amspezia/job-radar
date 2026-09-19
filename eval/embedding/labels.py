"""Label sources, views and effective-label resolution. Pure: no I/O, no DB."""

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

# Precedence order, highest first.
SOURCES = ("human_blind", "constructed", "llm_judge")
VIEWS = ("silver", "human", "effective")
HUMAN_SOURCES = ("human_blind", "constructed")  # what the `human` view counts

_RANK = {source: rank for rank, source in enumerate(SOURCES)}
_JUDGE = "llm_judge"
_VIEW_SOURCES = {
    "silver": frozenset({_JUDGE}),
    "human": frozenset(HUMAN_SOURCES),
    "effective": frozenset(SOURCES),
}


@dataclass(frozen=True)
class LabelRow:
    """One `embedding_label` row. `id` orders rows: larger means newer."""

    job_id: UUID
    grade: int
    source: str
    judge_run_id: UUID | None
    id: int


def resolve_effective(
    rows: Iterable[LabelRow], view: str, judge_run_id: UUID | None = None
) -> dict[UUID, int]:
    """Resolve one grade per job for `view`.

    Judge rows count only for `judge_run_id`, and views that include the judge require it:
    without the pin a spike run or a new prompt version would silently mix into a comparison.
    Within a source the largest `id` wins (`now()` is constant inside a transaction, so
    timestamps cannot order a batch); across sources the highest precedence wins, so a later
    human row supersedes an older judge one even when it lowers the grade.
    """
    if view not in _VIEW_SOURCES:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    counted = _VIEW_SOURCES[view]
    if _JUDGE in counted and judge_run_id is None:
        raise ValueError(f"view {view!r} includes llm_judge labels and requires a judge_run_id")

    best: dict[UUID, LabelRow] = {}
    for row in rows:
        if row.source not in _RANK:
            raise ValueError(f"unknown label source {row.source!r}; expected one of {SOURCES}")
        if not 0 <= row.grade <= 3:
            raise ValueError(f"grade {row.grade} for job {row.job_id} is outside 0..3")
        if row.source not in counted:
            continue
        if row.source == _JUDGE and row.judge_run_id != judge_run_id:
            continue
        current = best.get(row.job_id)
        if current is None or (_RANK[row.source], -row.id) < (_RANK[current.source], -current.id):
            best[row.job_id] = row
    return {job_id: row.grade for job_id, row in best.items()}


def is_partial(view: str) -> bool:
    """True when the view does not label every corpus job (only `human`)."""
    if view not in _VIEW_SOURCES:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    return view == "human"
