"""Build a judge for a topic, label the topic's pool with it, and calibrate the result.

`label_topic` is the only place the judge touches the database: workers grade preloaded
`JobView`s concurrently, and the collecting coroutine alone writes rows, in committed batches,
so a crash or a stop keeps everything graded so far and a resume picks up from there. A job the
judge cannot grade after retries is counted as failed; it never receives a default grade.
"""

import asyncio
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid4

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.checks import record_check
from eval.embedding.judge.base import CandidateBrief, JobView, Judge, JudgeError, Judgment
from eval.embedding.judge.calibration import (
    agreement,
    binary_prf,
    confusion_matrix,
    quadratic_weighted_kappa,
)
from eval.embedding.judge.fewshot import GRADES, select_fewshot, split_dev_test
from eval.embedding.judge.ollama_judge import ChatClient, OllamaJudge
from eval.embedding.labels import HUMAN_SOURCES, LabelRow, resolve_effective
from eval.embedding.models import (
    EmbeddingJob,
    EmbeddingJudgeRun,
    EmbeddingLabel,
    EmbeddingTopic,
    EmbeddingTopicJob,
)
from eval.llm.ollama import TagInfo

log = logging.getLogger(__name__)

BATCH_SIZE = 25
MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0
MIN_CALIBRATION_PAIRS = 10
_ORDER_SEED = 0  # the shuffle behind `limit`, and the dev/held-out split

_CAVEAT_LABELS = (
    "the labels are the user's own blind grades of a mixed pooled-and-random sample, not a "
    "uniform sample of the corpus"
)
_CAVEAT_VIEW = "the labels were graded on the 400-char label view, and the judge sees the same view"


def caveats(n: int) -> tuple[str, ...]:
    """What limits a calibration on `n` pairs; the standard error of kappa is stated for `n`."""
    return (
        _CAVEAT_LABELS,
        f"n={n} labeled pairs: the standard error of kappa is roughly 0.1 at n~46, so a kappa "
        "within about 0.1 of the threshold is not decisive",
        _CAVEAT_VIEW,
    )


_RETRYABLE = (JudgeError, httpx.TransportError)


class JudgeClient(ChatClient, Protocol):
    async def tags(self) -> dict[str, TagInfo]: ...


@dataclass
class CalibrationResult:
    kappa: float
    passed: bool
    threshold: float
    n: int
    exact: float
    within1: float
    binary: dict[str, float]
    confusion: list[list[int]]
    split: str
    n_excluded_exemplars: int
    judge_run_id: UUID


async def _topic_by_name(session: AsyncSession, topic_name: str) -> EmbeddingTopic:
    topic = (
        await session.execute(select(EmbeddingTopic).where(EmbeddingTopic.name == topic_name))
    ).scalar_one_or_none()
    if topic is None:
        raise LookupError(f"no topic named {topic_name!r}")
    return topic


async def _corpus_job_ids(session: AsyncSession, topic_id: UUID) -> set[UUID]:
    rows = await session.execute(
        select(EmbeddingTopicJob.job_id).where(EmbeddingTopicJob.topic_id == topic_id)
    )
    return set(rows.scalars())


async def _label_rows(session: AsyncSession, topic_id: UUID) -> list[LabelRow]:
    rows = await session.execute(
        select(
            EmbeddingLabel.job_id,
            EmbeddingLabel.grade,
            EmbeddingLabel.source,
            EmbeddingLabel.judge_run_id,
            EmbeddingLabel.id,
        ).where(EmbeddingLabel.topic_id == topic_id)
    )
    return [LabelRow(*row) for row in rows]


async def _human_grades(session: AsyncSession, topic_id: UUID) -> dict[UUID, int]:
    """The human view of the topic's labels, restricted to jobs in its corpus."""
    corpus = await _corpus_job_ids(session, topic_id)
    resolved = resolve_effective(await _label_rows(session, topic_id), "human")
    return {job_id: grade for job_id, grade in resolved.items() if job_id in corpus}


def _job_view(row) -> JobView:
    return JobView(
        title=row.title,
        company=row.company,
        source=row.source,
        seniority=row.seniority,
        requirements=row.requirements,
        responsibilities=row.responsibilities,
        description=row.description,
    )


_VIEW_COLUMNS = (
    EmbeddingJob.id,
    EmbeddingJob.title,
    EmbeddingJob.company,
    EmbeddingJob.source,
    EmbeddingJob.seniority,
    EmbeddingJob.requirements,
    EmbeddingJob.responsibilities,
    EmbeddingJob.description,
)


async def build_judge(
    session: AsyncSession,
    topic_name: str,
    client: JudgeClient,
    model: str,
    *,
    per_grade: int = 1,
    fewshot_seed: int = 13,
    seed: int = 1,
) -> tuple[OllamaJudge, list[UUID]]:
    """A judge briefed from the topic's profile snapshot, with seeded human-labeled exemplars.

    Exemplars come from the human view (`human_blind` / `constructed`), `per_grade` of each
    grade 0-3. Returns the judge and the exemplar job ids (which `calibrate` excludes).
    """
    topic = await _topic_by_name(session, topic_name)
    brief = CandidateBrief.from_snapshot(topic.profile_snapshot)
    human = await _human_grades(session, topic.id)
    short = [
        f"grade {grade} (has {have}, needs {per_grade})"
        for grade in sorted(GRADES)
        if (have := sum(g == grade for g in human.values())) < per_grade
    ]
    if short:
        raise ValueError(
            f"topic {topic_name!r} has too few human labels for the few-shot exemplars; "
            f"lacking {', '.join(short)}: label more jobs with `label-blind`"
        )
    fewshot_ids: list[UUID] = select_fewshot(human, per_grade, fewshot_seed)
    rows = await session.execute(select(*_VIEW_COLUMNS).where(EmbeddingJob.id.in_(fewshot_ids)))
    views = {row.id: _job_view(row) for row in rows}
    fewshot = [(views[job_id], human[job_id]) for job_id in fewshot_ids]

    judge = OllamaJudge(client, model, brief, fewshot, seed=seed)
    judge.model_digest = await _model_digest(client, model)  # type: ignore[attr-defined]
    return judge, fewshot_ids


async def _model_digest(client: JudgeClient, model: str) -> str | None:
    """Best effort: a run is still reproducible from its prompt and seed without the digest."""
    try:
        info = (await client.tags()).get(model)
    except Exception as exc:
        log.warning("could not read the digest of %s from ollama: %s", model, exc)
        return None
    return info.digest if info else None


async def _grade_with_retry(
    judge: Judge, job: JobView, backoff: float
) -> tuple[Judgment | None, str | None]:
    error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await judge.grade(job), None
        except _RETRYABLE as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(backoff * attempt)
    return None, error


async def _open_run(
    session: AsyncSession,
    topic_id: UUID,
    judge: Judge,
    fewshot_ids: Sequence[UUID],
    concurrency: int,
    limit: int | None,
    resume_run_id: UUID | None,
) -> UUID:
    description = judge.describe()
    if resume_run_id is not None:
        run = await session.get(EmbeddingJudgeRun, resume_run_id)
        if run is None or run.topic_id != topic_id:
            raise LookupError(f"judge run {resume_run_id} does not belong to topic {topic_id}")
        if (run.model, run.prompt_version, run.prompt_sha) != (
            description["model"],
            description["prompt_version"],
            description["prompt_sha"],
        ):
            raise ValueError(
                f"judge run {resume_run_id} was made with {run.model} / prompt "
                f"{run.prompt_version} ({run.prompt_sha[:8]}); resuming with a different "
                "judge would mix two prompts in one run"
            )
        await session.execute(
            update(EmbeddingJudgeRun)
            .where(EmbeddingJudgeRun.id == resume_run_id)
            .values(status="running")
        )
        await session.commit()
        return resume_run_id

    human = await _human_grades(session, topic_id)
    missing = [str(job_id) for job_id in fewshot_ids if job_id not in human]
    if missing:
        raise ValueError(f"few-shot exemplars without a human label: {missing}")
    run = EmbeddingJudgeRun(
        id=uuid4(),
        topic_id=topic_id,
        model=description["model"],
        model_digest=getattr(judge, "model_digest", None),
        prompt_version=description["prompt_version"],
        prompt_sha=description["prompt_sha"],
        params={**description, "concurrency": concurrency, "limit": limit},
        fewshot=[{"job_id": str(job_id), "grade": human[job_id]} for job_id in fewshot_ids],
        status="running",
    )
    session.add(run)
    await session.commit()
    return run.id


async def _flush(session: AsyncSession, pending: list[EmbeddingLabel]) -> None:
    if pending:
        session.add_all(pending)
        await session.commit()
        pending.clear()


async def label_topic(
    session: AsyncSession,
    topic_id: UUID,
    judge: Judge,
    fewshot_ids: Sequence[UUID],
    concurrency: int = 4,
    limit: int | None = None,
    resume_run_id: UUID | None = None,
    *,
    labeled_only: bool = False,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
) -> UUID:
    """Label the topic's corpus with `judge`, resumably; returns the judge run id.

    Every corpus job without an `llm_judge` label in the run is graded (exemplars included),
    in a seeded shuffle of the id-sorted list so `limit` draws a reproducible random sample.
    `labeled_only` restricts the work to jobs a human has graded: a few minutes instead of an
    hour, which is what iterating a prompt against the human labels needs. The run is an
    ordinary judge run, so `resume_run_id` later completes the rest of the pool.
    """
    if await session.get(EmbeddingTopic, topic_id) is None:
        raise LookupError(f"no topic with id {topic_id}")
    run_id = await _open_run(
        session, topic_id, judge, fewshot_ids, concurrency, limit, resume_run_id
    )
    started = time.monotonic()
    pending: list[EmbeddingLabel] = []
    n_failed = 0
    try:
        views = {
            row.id: _job_view(row)
            for row in await session.execute(
                select(*_VIEW_COLUMNS)
                .join(EmbeddingTopicJob, EmbeddingTopicJob.job_id == EmbeddingJob.id)
                .where(EmbeddingTopicJob.topic_id == topic_id)
            )
        }
        done = set(
            (
                await session.execute(
                    select(EmbeddingLabel.job_id).where(
                        EmbeddingLabel.judge_run_id == run_id,
                        EmbeddingLabel.source == "llm_judge",
                    )
                )
            ).scalars()
        )
        order = sorted(views, key=str)
        random.Random(_ORDER_SEED).shuffle(order)
        if labeled_only:
            human = set(
                (
                    await session.execute(
                        select(EmbeddingLabel.job_id).where(
                            EmbeddingLabel.topic_id == topic_id,
                            EmbeddingLabel.source.in_(HUMAN_SOURCES),
                        )
                    )
                ).scalars()
            )
            order = [job_id for job_id in order if job_id in human]
        todo = [job_id for job_id in order if job_id not in done][:limit]

        gate = asyncio.Semaphore(concurrency)

        async def work(job_id: UUID) -> tuple[UUID, Judgment | None, str | None]:
            async with gate:
                judgment, error = await _grade_with_retry(judge, views[job_id], backoff)
            return job_id, judgment, error

        tasks = [asyncio.create_task(work(job_id)) for job_id in todo]
        try:
            for processed, finished in enumerate(asyncio.as_completed(tasks), start=1):
                job_id, judgment, error = await finished
                if judgment is None:
                    n_failed += 1
                    log.warning(
                        "judge failed on job %s after %d attempts: %s", job_id, MAX_ATTEMPTS, error
                    )
                else:
                    pending.append(
                        EmbeddingLabel(
                            topic_id=topic_id,
                            job_id=job_id,
                            grade=judgment.grade,
                            source="llm_judge",
                            judge_run_id=run_id,
                            rationale=judgment.rationale,
                        )
                    )
                if len(pending) >= BATCH_SIZE:
                    await _flush(session, pending)
                if processed % BATCH_SIZE == 0:
                    log.info(
                        "judge run %s: %d/%d graded, %d failed, %.1f s elapsed",
                        run_id,
                        processed,
                        len(todo),
                        n_failed,
                        time.monotonic() - started,
                    )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await _flush(session, pending)
    except BaseException:
        await session.rollback()  # a broken transaction must not block keeping finished work
        await _flush(session, pending)
        await _finish_run(session, run_id, started, "failed", n_failed)
        raise
    await _finish_run(session, run_id, started, "completed", n_failed)
    return run_id


async def _finish_run(
    session: AsyncSession, run_id: UUID, started: float, status: str, n_failed: int
) -> None:
    n_labeled = (
        await session.execute(
            select(func.count(func.distinct(EmbeddingLabel.job_id))).where(
                EmbeddingLabel.judge_run_id == run_id, EmbeddingLabel.source == "llm_judge"
            )
        )
    ).scalar_one()
    previous = (
        await session.execute(
            select(EmbeddingJudgeRun.seconds).where(EmbeddingJudgeRun.id == run_id)
        )
    ).scalar_one()
    await session.execute(
        update(EmbeddingJudgeRun)
        .where(EmbeddingJudgeRun.id == run_id)
        .values(
            n_labeled=n_labeled,
            n_failed=n_failed,
            seconds=(previous or 0.0) + time.monotonic() - started,
            status=status,
        )
    )
    await session.commit()


async def calibrate(
    session: AsyncSession,
    topic_name: str,
    judge_run_id: UUID,
    min_kappa: float = 0.6,
    use_split: bool = False,
) -> CalibrationResult:
    """Judge-vs-human agreement for one run, recorded as a `judge_calibration` check.

    Compared on the jobs that carry both a human-view grade (`human_blind` / `constructed`) and
    a grade from this run, minus the run's few-shot exemplars (the judge was shown their
    grades): all of them by default. With `use_split` only the held-out half is used, so the
    prompt can be iterated on the other half.
    """
    topic = await _topic_by_name(session, topic_name)
    run = await session.get(EmbeddingJudgeRun, judge_run_id)
    if run is None or run.topic_id != topic.id:
        raise LookupError(f"judge run {judge_run_id} does not belong to topic {topic_name!r}")

    corpus = await _corpus_job_ids(session, topic.id)
    rows = await _label_rows(session, topic.id)
    human = {j: g for j, g in resolve_effective(rows, "human").items() if j in corpus}
    judged = {
        j: g for j, g in resolve_effective(rows, "silver", judge_run_id).items() if j in corpus
    }
    exemplars = {UUID(item["job_id"]) for item in run.fewshot}

    paired = human.keys() & judged.keys()
    n_excluded = len(paired & exemplars)
    ids = sorted(paired - exemplars, key=str)
    if use_split:
        _, ids = split_dev_test(ids, seed=_ORDER_SEED)
    split = "held-out half" if use_split else "all non-exemplar"
    if len(ids) < MIN_CALIBRATION_PAIRS:
        raise ValueError(
            f"only {len(ids)} human-and-judge pairs on the {split} of run {judge_run_id} "
            f"(need at least {MIN_CALIBRATION_PAIRS}); label more jobs with `label-blind` first"
        )

    y_true = [human[job_id] for job_id in ids]
    y_pred = [judged[job_id] for job_id in ids]
    kappa = quadratic_weighted_kappa(y_true, y_pred)
    result = CalibrationResult(
        kappa=kappa,
        passed=kappa >= min_kappa,
        threshold=min_kappa,
        n=len(ids),
        exact=agreement(y_true, y_pred)["exact"],
        within1=agreement(y_true, y_pred)["within1"],
        binary=binary_prf(y_true, y_pred),
        confusion=confusion_matrix(y_true, y_pred),
        split=split,
        n_excluded_exemplars=n_excluded,
        judge_run_id=judge_run_id,
    )
    await record_check(
        session,
        topic.id,
        "judge_calibration",
        result.passed,
        {
            "kappa": result.kappa,
            "threshold": result.threshold,
            "n": result.n,
            "exact": result.exact,
            "within1": result.within1,
            "binary": result.binary,
            "confusion": result.confusion,
            "split": result.split,
            "n_excluded_exemplars": result.n_excluded_exemplars,
            "judge_run_id": str(judge_run_id),
            "caveats": list(caveats(result.n)),
        },
        judge_run_id=judge_run_id,
    )
    return result
