"""Recorded pass/fail checks that gate a comparison. No `job_radar` imports."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.models import EmbeddingCheck

# What `verify` records and `evaluate` insists on before it will trust a report.
REQUIRED_CHECKS = ("parity_ranking", "parity_reconstruction")

_CALIBRATION = "judge_calibration"


@dataclass(frozen=True)
class CheckRow:
    id: int
    name: str
    passed: bool
    detail: dict
    judge_run_id: UUID | None
    created_at: datetime


def _row(check: EmbeddingCheck) -> CheckRow:
    return CheckRow(
        id=check.id,
        name=check.name,
        passed=check.passed,
        detail=check.detail,
        judge_run_id=check.judge_run_id,
        created_at=check.created_at,
    )


async def record_check(
    session: AsyncSession,
    topic_id: UUID,
    name: str,
    passed: bool,
    detail: dict,
    judge_run_id: UUID | None = None,
) -> None:
    """Append a check result. Rows are never updated: a re-run adds a newer one."""
    session.add(
        EmbeddingCheck(
            topic_id=topic_id,
            name=name,
            passed=passed,
            detail=detail,
            judge_run_id=judge_run_id,
        )
    )
    await session.commit()


async def latest_checks(session: AsyncSession, topic_id: UUID) -> dict[str, CheckRow]:
    """The newest row (largest `id`; `now()` is constant inside a transaction) per check name."""
    rows = await session.scalars(
        select(EmbeddingCheck)
        .where(EmbeddingCheck.topic_id == topic_id)
        .order_by(EmbeddingCheck.id)
    )
    return {check.name: _row(check) for check in rows}


async def calibration_for_run(
    session: AsyncSession, topic_id: UUID, judge_run_id: UUID
) -> CheckRow | None:
    """The newest `judge_calibration` recorded for this judge run, or None if never calibrated."""
    check = await session.scalar(
        select(EmbeddingCheck)
        .where(
            EmbeddingCheck.topic_id == topic_id,
            EmbeddingCheck.name == _CALIBRATION,
            EmbeddingCheck.judge_run_id == judge_run_id,
        )
        .order_by(EmbeddingCheck.id.desc())
        .limit(1)
    )
    return None if check is None else _row(check)
