"""Blind labeling: the user grades postings without seeing any model output, and only those
`human_blind` labels are gold.

Candidates are the topic's documents that carry no human label yet, in an order fixed by
`seed` alone: the order is computed over the whole corpus and the labeled documents are
dropped afterwards, so re-running with the same seed continues where the last session stopped.
`pooled` documents are the ones the primary runs disagree on (in the top 30 of some but not
all of them), the ones a ranking comparison is decided by; `random` documents are the rest
(every document when the topic has no runs to pool from).
Only documents in `language` (English by default, `None` for all) are offered; as with labeled
documents, the others are dropped after the ordering, so the order of the rest does not depend
on the filter.
Nothing about a document's runs, ranks, scores or bucket is ever shown.

No `job_radar` imports.
"""

import random
from collections import Counter, deque
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.judge.base import JobView
from eval.embedding.judge.prompt import render_label_view
from eval.embedding.labels import HUMAN_SOURCES
from eval.embedding.language import doc_language
from eval.embedding.models import (
    EmbeddingJob,
    EmbeddingLabel,
    EmbeddingRun,
    EmbeddingRunRanking,
    EmbeddingTopic,
    EmbeddingTopicJob,
)

BUCKETS = ("mixed", "pooled", "random")
POOL_DEPTH = 30
_POOLED_PER_BLOCK, _RANDOM_PER_BLOCK = 3, 2  # the ~60% / ~40% of a mixed ordering
_GRADE_PROMPT = "Grade [0-3 | s=skip | q=quit]: "
_NOTE_PROMPT = "Note (Enter = none): "

_HEADER = (
    "Grade each posting by how relevant it is to the job search you are running.",
    "  3 Strong        exactly the role you want; right stack, level and domain",
    "  2 Relevant      you would apply; maybe one thing is off",
    "  1 Marginal      adjacent role, off-by-one seniority, or a key skill missing",
    "  0 Not relevant  wrong stack or domain, or clearly off target",
    "Location is not shown; judge role/stack fit from what you see.",
)


async def _pooled_ids(session: AsyncSession, topic_id: UUID) -> set[UUID] | None:
    """Documents in the top `POOL_DEPTH` of some but not all latest completed primary runs.

    None when the topic has fewer than two such runs, so no disagreement can be measured.
    """
    runs = await session.scalars(
        select(EmbeddingRun)
        .where(EmbeddingRun.topic_id == topic_id, EmbeddingRun.status == "completed")
        .order_by(EmbeddingRun.started_at, EmbeddingRun.id)
    )
    latest = {run.name: run.id for run in runs if ":" not in run.name}
    if len(latest) < 2:
        return None
    hits: Counter[UUID] = Counter()
    for run_id in latest.values():
        top = await session.scalars(
            select(EmbeddingRunRanking.job_id)
            .where(EmbeddingRunRanking.run_id == run_id)
            .order_by(EmbeddingRunRanking.rank)
            .limit(POOL_DEPTH)
        )
        hits.update(top)
    return {job_id for job_id, count in hits.items() if count < len(latest)}


def _mixed(pooled: list[UUID], rest: list[UUID], rnd: random.Random) -> list[UUID]:
    """Blocks of three pooled and two random, each block shuffled; the longer tail follows."""
    out: list[UUID] = []
    p = r = 0
    while p < len(pooled) and r < len(rest):
        block = pooled[p : p + _POOLED_PER_BLOCK] + rest[r : r + _RANDOM_PER_BLOCK]
        p += _POOLED_PER_BLOCK
        r += _RANDOM_PER_BLOCK
        rnd.shuffle(block)
        out += block
    return out + pooled[p:] + rest[r:]


async def _in_language(session: AsyncSession, topic_id: UUID, language: str) -> set[UUID]:
    rows = await session.execute(
        select(
            EmbeddingJob.id,
            EmbeddingJob.title,
            EmbeddingJob.description,
            EmbeddingJob.requirements,
            EmbeddingJob.responsibilities,
        )
        .join(EmbeddingTopicJob, EmbeddingTopicJob.job_id == EmbeddingJob.id)
        .where(EmbeddingTopicJob.topic_id == topic_id)
    )
    return {job_id for job_id, *fields in rows if doc_language(*fields) == language}


async def _candidates(
    session: AsyncSession, topic_id: UUID, bucket: str, seed: int, language: str | None = None
) -> tuple[list[UUID], dict[UUID, str]]:
    """The unlabeled documents in `language` in session order, and every document's stratum."""
    corpus = sorted(
        await session.scalars(
            select(EmbeddingTopicJob.job_id).where(EmbeddingTopicJob.topic_id == topic_id)
        ),
        key=str,
    )
    labeled = set(
        await session.scalars(
            select(EmbeddingLabel.job_id).where(
                EmbeddingLabel.topic_id == topic_id, EmbeddingLabel.source.in_(HUMAN_SOURCES)
            )
        )
    )
    pooled_set = await _pooled_ids(session, topic_id)
    if pooled_set is None:
        if bucket != "random":
            raise ValueError(
                f"the {bucket!r} bucket needs at least two completed primary runs on the topic "
                "to find the postings they disagree on: run the embedders first"
            )
        pooled_set = set()
    pooled = [j for j in corpus if j in pooled_set]
    rest = [j for j in corpus if j not in pooled_set]
    rnd = random.Random(seed)
    rnd.shuffle(pooled)
    rnd.shuffle(rest)
    strata = {**dict.fromkeys(pooled, "pooled"), **dict.fromkeys(rest, "random")}
    if bucket == "pooled":
        ordered = pooled
    elif bucket == "random":
        ordered = rest
    else:
        ordered = _mixed(pooled, rest, rnd)
    offered = None if language is None else await _in_language(session, topic_id, language)
    return [j for j in ordered if j not in labeled and (offered is None or j in offered)], strata


async def _job_view(session: AsyncSession, job_id: UUID) -> JobView:
    row = (
        await session.execute(
            select(
                EmbeddingJob.title,
                EmbeddingJob.company,
                EmbeddingJob.source,
                EmbeddingJob.seniority,
                EmbeddingJob.requirements,
                EmbeddingJob.responsibilities,
                EmbeddingJob.description,
            ).where(EmbeddingJob.id == job_id)
        )
    ).one()
    return JobView(*row)


def _ask(input_fn: Callable[[str], str], prompt: str) -> str | None:
    """One line of input, or None when the user is done (end of input or Ctrl-C)."""
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _ask_grade(input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> int | str:
    """A grade 0-3, `"s"` (skip) or `"q"` (quit, also for end of input); else ask again."""
    while True:
        raw = _ask(input_fn, _GRADE_PROMPT)
        if raw is None:
            return "q"
        answer = raw.strip().lower()
        if answer in ("s", "q"):
            return answer
        if answer in ("0", "1", "2", "3"):
            return int(answer)
        output_fn("enter 0, 1, 2, 3, s or q")


async def label_blind(
    session: AsyncSession,
    topic_name: str,
    n: int = 50,
    bucket: str = "mixed",
    seed: int = 0,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    language: str | None = "en",
) -> dict:
    """Ask for up to `n` grades and store each as a `human_blind` label the moment it is given.

    `s` skips a posting without writing anything: it is offered once more after the rest of the
    session's postings, and in a later session at its seeded position. `q`, end of input and
    Ctrl-C end the session; every grade already given is committed.
    Returns `{"labeled", "skipped", "remaining", "buckets"}`; `remaining` counts the postings of
    the chosen bucket (and `language`) that are still unlabeled.
    """
    if bucket not in BUCKETS:
        raise ValueError(f"unknown bucket {bucket!r}; expected one of {BUCKETS}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    topic_id = await session.scalar(
        select(EmbeddingTopic.id).where(EmbeddingTopic.name == topic_name)
    )
    if topic_id is None:
        raise LookupError(f"no topic named {topic_name!r}")

    unlabeled, strata = await _candidates(session, topic_id, bucket, seed, language)
    total = min(n, len(unlabeled))
    if total == 0:
        output_fn("nothing left to label")
        return {"labeled": 0, "skipped": 0, "remaining": 0, "buckets": {}}
    for line in _HEADER:
        output_fn(line)

    queue = deque(unlabeled)
    skipped: set[UUID] = set()
    labeled: Counter[str] = Counter()
    while queue and labeled.total() < n:
        job_id = queue.popleft()
        output_fn(f"\n[{labeled.total() + 1}/{total}]")
        output_fn(render_label_view(await _job_view(session, job_id)))
        answer = _ask_grade(input_fn, output_fn)
        if answer == "q":
            break
        if answer == "s":
            if job_id not in skipped:
                skipped.add(job_id)
                queue.append(job_id)
            continue
        note = _ask(input_fn, _NOTE_PROMPT)
        rationale = f"bucket={strata[job_id]}"
        if note and note.strip():
            rationale += f"; {note.strip()}"
        session.add(
            EmbeddingLabel(
                topic_id=topic_id,
                job_id=job_id,
                grade=int(answer),
                source="human_blind",
                rationale=rationale,
            )
        )
        await session.commit()
        skipped.discard(job_id)
        labeled[strata[job_id]] += 1
        if note is None:
            break

    return {
        "labeled": labeled.total(),
        "skipped": len(skipped),
        "remaining": len(unlabeled) - labeled.total(),
        "buckets": dict(labeled),
    }
