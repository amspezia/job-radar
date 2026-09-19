import asyncio
import hashlib
from collections import Counter
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import select

from eval.embedding.checks import calibration_for_run
from eval.embedding.judge.base import JobView, JudgeError, Judgment
from eval.embedding.judge.fewshot import split_dev_test
from eval.embedding.judge.ollama_judge import OllamaJudge
from eval.embedding.judge.runner import (
    BATCH_SIZE,
    CalibrationResult,
    build_judge,
    calibrate,
    label_topic,
)
from eval.embedding.models import (
    EmbeddingJob,
    EmbeddingJudgeRun,
    EmbeddingLabel,
    EmbeddingTopic,
    EmbeddingTopicJob,
)
from eval.llm.ollama import TagInfo

SNAPSHOT = {
    "seniority": "senior",
    "years_experience": 8,
    "target_titles": ["Platform Engineer"],
    "tech_stack": ["python", "postgres"],
    "domains": ["infra"],
    "remote_required": True,
    "cv_text": "scrubbed cv text",
}


def _title_grade(job: JobView) -> int:
    return int(hashlib.sha1(job.title.encode()).hexdigest(), 16) % 4


def _index(job: JobView) -> int:
    return int(job.title.removeprefix("job-"))


def _index_grade(job: JobView) -> int:
    return _index(job) % 4


class FakeJudge:
    """Grades from the title; scripted failures; tracks how many calls run at once."""

    name = "fake"

    def __init__(
        self,
        grade_fn=_title_grade,
        failures: dict[str, int] | None = None,
        error: type[Exception] = JudgeError,
        fatal_after: int | None = None,
        delay: float = 0.0,
        prompt_sha: str = "sha-1",
    ) -> None:
        self._grade_fn = grade_fn
        self._failures = dict(failures or {})
        self._error = error
        self._fatal_after = fatal_after
        self._delay = delay
        self._prompt_sha = prompt_sha
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def describe(self) -> dict:
        return {
            "model": "fake-judge",
            "prompt_version": "t1",
            "prompt_sha": self._prompt_sha,
            "seed": 1,
        }

    async def grade(self, job: JobView) -> Judgment:
        self.calls.append(job.title)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delay)
            if self._fatal_after is not None and len(self.calls) > self._fatal_after:
                raise RuntimeError("boom")
            if self._failures.get(job.title, 0) > 0:
                self._failures[job.title] -= 1
                raise self._error(f"scripted failure for {job.title}")
            return Judgment(grade=self._grade_fn(job), rationale=f"because {job.title}")
        finally:
            self.in_flight -= 1


class FakeChatClient:
    def __init__(self, digest: str | None = "sha256:abc") -> None:
        self._digest = digest
        self.messages: list[list[dict]] = []

    async def chat_json(self, model, messages, schema, *, options):
        self.messages.append(messages)
        return SimpleNamespace(data={"reason": "ok", "grade": 1}, prompt_eval_seconds=None)

    async def tags(self):
        if self._digest is None:
            raise httpx.ConnectError("ollama is down")
        return {"gemma": TagInfo(digest=self._digest, quantization=None, context_length=None)}


async def _seed(session, n_jobs: int, human=lambda i: i % 4, name: str = "topic"):
    """A topic of `n_jobs` postings titled job-000...; `human(i)` is the blind grade or None."""
    topic = EmbeddingTopic(
        id=uuid4(),
        tier="A",
        name=name,
        status="frozen",
        profile_snapshot=SNAPSHOT,
        query_inputs={},
        builder="test",
    )
    session.add(topic)
    jobs = []
    for i in range(n_jobs):
        jobs.append(
            EmbeddingJob(
                id=uuid4(),
                origin="test",
                origin_id=f"{name}-{i}",
                source="himalayas",
                title=f"job-{i:03d}",
                company="Acme",
                url=f"https://example.com/{name}/{i}",
                description="desc",
                embed_text="text",
                embed_text_sha=f"sha-{name}-{i}",
            )
        )
    session.add_all(jobs)
    await session.flush()
    session.add_all(EmbeddingTopicJob(topic_id=topic.id, job_id=job.id) for job in jobs)
    for i, job in enumerate(jobs):
        grade = human(i)
        if grade is not None:
            session.add(
                EmbeddingLabel(topic_id=topic.id, job_id=job.id, grade=grade, source="human_blind")
            )
    await session.commit()
    return topic.id, [job.id for job in jobs]


async def _judge_labels(session, run_id) -> list[EmbeddingLabel]:
    rows = await session.execute(
        select(EmbeddingLabel).where(EmbeddingLabel.judge_run_id == run_id)
    )
    return list(rows.scalars())


async def _run(session, run_id) -> EmbeddingJudgeRun:
    run = await session.get(EmbeddingJudgeRun, run_id)
    await session.refresh(run)
    return run


# --- build_judge ---------------------------------------------------------------------------


async def test_build_judge_draws_one_exemplar_per_grade_from_human_labels_only(evals_session):
    topic_id, job_ids = await _seed(evals_session, 24)
    # A judge label for the same jobs must not leak into the exemplars.
    evals_session.add(
        EmbeddingLabel(topic_id=topic_id, job_id=job_ids[0], grade=3, source="llm_judge")
    )
    await evals_session.commit()
    client = FakeChatClient()

    judge, fewshot_ids = await build_judge(evals_session, "topic", client, "gemma")

    assert isinstance(judge, OllamaJudge)
    assert len(fewshot_ids) == 4 == len(set(fewshot_ids))
    index = {job_id: i for i, job_id in enumerate(job_ids)}
    assert Counter(index[j] % 4 for j in fewshot_ids) == {0: 1, 1: 1, 2: 1, 3: 1}
    await judge.grade(JobView("t", "c", "s", None, None, None, "d"))
    system = client.messages[0][0]["content"]
    assert "Platform Engineer" in system  # the brief comes from the profile snapshot
    assert all(f"job-{index[j]:03d}" in system for j in fewshot_ids)
    assert judge.model_digest == "sha256:abc"


async def test_build_judge_per_grade_is_configurable(evals_session):
    await _seed(evals_session, 24)

    _, fewshot_ids = await build_judge(
        evals_session, "topic", FakeChatClient(), "gemma", per_grade=2
    )

    assert len(fewshot_ids) == 8


async def test_build_judge_counts_constructed_labels_as_human(evals_session):
    topic_id, job_ids = await _seed(evals_session, 8, human=lambda i: None)
    evals_session.add_all(
        EmbeddingLabel(topic_id=topic_id, job_id=job_id, grade=i % 4, source="constructed")
        for i, job_id in enumerate(job_ids)
    )
    await evals_session.commit()

    _, fewshot_ids = await build_judge(evals_session, "topic", FakeChatClient(), "gemma")

    assert len(fewshot_ids) == 4


async def test_build_judge_names_the_grades_that_lack_labels(evals_session):
    await _seed(evals_session, 20, human=lambda i: i % 2 if i < 10 else None)  # only 0 and 1

    with pytest.raises(
        ValueError, match=r"lacking grade 2 \(has 0, needs 1\), grade 3 \(has 0, needs 1\)"
    ) as raised:
        await build_judge(evals_session, "topic", FakeChatClient(), "gemma")
    assert "label-blind" in str(raised.value)


async def test_build_judge_ignores_llm_labels_when_counting_exemplars(evals_session):
    topic_id, job_ids = await _seed(evals_session, 8, human=lambda i: None)
    evals_session.add_all(
        EmbeddingLabel(topic_id=topic_id, job_id=job_id, grade=i % 4, source="llm_judge")
        for i, job_id in enumerate(job_ids)
    )
    await evals_session.commit()

    with pytest.raises(ValueError, match="label-blind"):
        await build_judge(evals_session, "topic", FakeChatClient(), "gemma")


async def test_build_judge_is_deterministic_per_seed(evals_session):
    await _seed(evals_session, 40)
    client = FakeChatClient()

    first = (await build_judge(evals_session, "topic", client, "gemma", fewshot_seed=5))[1]
    again = (await build_judge(evals_session, "topic", client, "gemma", fewshot_seed=5))[1]
    others = [
        (await build_judge(evals_session, "topic", client, "gemma", fewshot_seed=s))[1]
        for s in (6, 7, 8)
    ]

    assert first == again
    assert any(other != first for other in others)


async def test_build_judge_keeps_going_without_a_digest(evals_session):
    await _seed(evals_session, 16)

    judge, _ = await build_judge(evals_session, "topic", FakeChatClient(digest=None), "gemma")

    assert judge.model_digest is None


async def test_build_judge_rejects_an_unknown_topic(evals_session):
    with pytest.raises(LookupError):
        await build_judge(evals_session, "nope", FakeChatClient(), "gemma")


# --- label_topic ---------------------------------------------------------------------------


async def test_label_topic_labels_every_job_once_and_records_the_run(evals_session):
    topic_id, job_ids = await _seed(evals_session, 30)
    exemplars = [job_ids[3], job_ids[6]]  # human grades 3 and 2
    judge = FakeJudge()
    judge.model_digest = "sha256:abc"

    run_id = await label_topic(
        evals_session, topic_id, judge, exemplars, concurrency=3, limit=None, backoff=0
    )

    labels = await _judge_labels(evals_session, run_id)
    assert sorted(label.job_id for label in labels) == sorted(job_ids)  # exemplars included
    assert {label.source for label in labels} == {"llm_judge"}
    assert {label.topic_id for label in labels} == {topic_id}
    assert all(label.rationale.startswith("because job-") for label in labels)
    run = await _run(evals_session, run_id)
    assert (run.topic_id, run.status, run.n_labeled, run.n_failed) == (topic_id, "completed", 30, 0)
    assert (run.model, run.model_digest) == ("fake-judge", "sha256:abc")
    assert (run.prompt_version, run.prompt_sha) == ("t1", "sha-1")
    assert run.params == {**judge.describe(), "concurrency": 3, "limit": None}
    assert run.fewshot == [
        {"job_id": str(exemplars[0]), "grade": 3},
        {"job_id": str(exemplars[1]), "grade": 2},
    ]
    assert run.seconds is not None and run.seconds >= 0


async def test_label_topic_limit_draws_a_reproducible_sample(evals_session):
    topic_id, job_ids = await _seed(evals_session, 40)

    first = await label_topic(evals_session, topic_id, FakeJudge(), [], limit=12, backoff=0)
    second = await label_topic(evals_session, topic_id, FakeJudge(), [], limit=12, backoff=0)

    sample = {label.job_id for label in await _judge_labels(evals_session, first)}
    assert sample == {label.job_id for label in await _judge_labels(evals_session, second)}
    assert len(sample) == 12
    assert sample != set(sorted(job_ids, key=str)[:12])  # shuffled, not the first ids


async def test_label_topic_labeled_only_grades_just_the_human_labeled_jobs(evals_session):
    human = lambda i: i % 4 if i < 6 else None  # noqa: E731
    topic_id, job_ids = await _seed(evals_session, 30, human=human)

    run_id = await label_topic(
        evals_session, topic_id, FakeJudge(), [], labeled_only=True, backoff=0
    )

    graded = {label.job_id for label in await _judge_labels(evals_session, run_id)}
    assert graded == set(job_ids[:6])
    assert (await _run(evals_session, run_id)).n_labeled == 6

    # An ordinary judge run: resuming without the flag finishes the rest of the pool.
    await label_topic(evals_session, topic_id, FakeJudge(), [], resume_run_id=run_id, backoff=0)
    assert len(await _judge_labels(evals_session, run_id)) == 30


async def test_label_topic_resume_skips_done_jobs_and_retries_failures(evals_session):
    topic_id, job_ids = await _seed(evals_session, 30)
    stubborn = FakeJudge(failures={"job-004": 99, "job-017": 99})
    run_id = await label_topic(evals_session, topic_id, stubborn, [], backoff=0)

    partial = await _judge_labels(evals_session, run_id)
    assert len(partial) == 28
    assert Counter(stubborn.calls)["job-004"] == 3  # tried three times, then counted
    failed_run = await _run(evals_session, run_id)
    assert (failed_run.n_labeled, failed_run.n_failed, failed_run.status) == (28, 2, "completed")

    healthy = FakeJudge()
    resumed = await label_topic(
        evals_session, topic_id, healthy, [], backoff=0, resume_run_id=run_id
    )

    assert resumed == run_id
    assert sorted(healthy.calls) == ["job-004", "job-017"]  # only what was missing
    labels = await _judge_labels(evals_session, run_id)
    assert sorted(label.job_id for label in labels) == sorted(job_ids)
    run = await _run(evals_session, run_id)
    assert (run.n_labeled, run.n_failed, run.status) == (30, 0, "completed")
    assert run.params["concurrency"] == 4  # the resume did not rewrite the run's parameters


async def test_label_topic_writes_no_default_grade_for_a_failed_job(evals_session):
    topic_id, _ = await _seed(evals_session, 12)
    judge = FakeJudge(failures={"job-005": 99})

    run_id = await label_topic(evals_session, topic_id, judge, [], backoff=0)

    labeled_titles = set(
        (
            await evals_session.execute(
                select(EmbeddingJob.title)
                .join(EmbeddingLabel, EmbeddingLabel.job_id == EmbeddingJob.id)
                .where(EmbeddingLabel.judge_run_id == run_id)
            )
        ).scalars()
    )
    assert "job-005" not in labeled_titles
    assert len(labeled_titles) == 11
    assert (await _run(evals_session, run_id)).n_failed == 1


@pytest.mark.parametrize("error", [JudgeError, httpx.ConnectError])
async def test_label_topic_retries_a_transient_error_then_succeeds(evals_session, error):
    topic_id, _ = await _seed(evals_session, 6)
    judge = FakeJudge(failures={"job-002": 2}, error=error)

    run_id = await label_topic(evals_session, topic_id, judge, [], backoff=0)

    assert Counter(judge.calls)["job-002"] == 3
    run = await _run(evals_session, run_id)
    assert (run.n_labeled, run.n_failed) == (6, 0)


async def test_label_topic_never_exceeds_the_concurrency_cap(evals_session):
    topic_id, _ = await _seed(evals_session, 16)
    judge = FakeJudge(delay=0.01)

    await label_topic(evals_session, topic_id, judge, [], concurrency=3, backoff=0)

    assert judge.max_in_flight == 3


async def test_label_topic_keeps_finished_batches_when_the_judge_breaks(evals_session):
    n_done = BATCH_SIZE + 5
    topic_id, _ = await _seed(evals_session, n_done + 10)
    judge = FakeJudge(fatal_after=n_done)

    with pytest.raises(RuntimeError, match="boom"):
        await label_topic(evals_session, topic_id, judge, [], concurrency=1, backoff=0)

    run_id = (await evals_session.execute(select(EmbeddingJudgeRun.id))).scalar_one()
    assert len(await _judge_labels(evals_session, run_id)) == n_done
    run = await _run(evals_session, run_id)
    assert (run.status, run.n_labeled, run.n_failed) == ("failed", n_done, 0)


async def test_label_topic_rejects_a_resume_of_another_topics_run(evals_session):
    topic_id, _ = await _seed(evals_session, 4)
    other_id, _ = await _seed(evals_session, 4, name="other")
    other_run = await label_topic(evals_session, other_id, FakeJudge(), [], backoff=0)

    with pytest.raises(LookupError):
        await label_topic(
            evals_session, topic_id, FakeJudge(), [], backoff=0, resume_run_id=other_run
        )
    with pytest.raises(LookupError):
        await label_topic(
            evals_session, topic_id, FakeJudge(), [], backoff=0, resume_run_id=uuid4()
        )


async def test_label_topic_rejects_resuming_with_a_different_prompt(evals_session):
    topic_id, _ = await _seed(evals_session, 4)
    run_id = await label_topic(evals_session, topic_id, FakeJudge(), [], backoff=0, limit=2)

    with pytest.raises(ValueError, match="different"):
        await label_topic(
            evals_session,
            topic_id,
            FakeJudge(prompt_sha="sha-2"),
            [],
            backoff=0,
            resume_run_id=run_id,
        )


# --- calibrate -----------------------------------------------------------------------------


async def _labeled(session, n_jobs: int, judge, exemplar_idx=(), human=lambda i: i % 4):
    topic_id, job_ids = await _seed(session, n_jobs, human=human)
    exemplars = [job_ids[i] for i in exemplar_idx]
    run_id = await label_topic(session, topic_id, judge, exemplars, backoff=0)
    return topic_id, job_ids, run_id


async def test_calibrate_perfect_agreement_passes_and_records_the_check(evals_session):
    topic_id, _, run_id = await _labeled(evals_session, 40, FakeJudge(grade_fn=_index_grade))

    result = await calibrate(evals_session, "topic", run_id)

    assert isinstance(result, CalibrationResult)
    assert result.kappa == pytest.approx(1.0)
    assert result.passed and result.threshold == 0.6
    assert (result.n, result.exact, result.within1) == (40, 1.0, 1.0)
    assert result.binary["f1"] == 1.0
    assert result.confusion == [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]
    assert (result.split, result.judge_run_id) == ("all non-exemplar", run_id)
    check = await calibration_for_run(evals_session, topic_id, run_id)
    assert check is not None and check.passed and check.judge_run_id == run_id
    assert check.detail["kappa"] == pytest.approx(1.0)
    assert (check.detail["threshold"], check.detail["n"]) == (0.6, 40)
    assert check.detail["confusion"] == result.confusion
    assert check.detail["split"] == "all non-exemplar"
    caveats = check.detail["caveats"]
    assert len(caveats) == 3
    assert any("blind grades" in c for c in caveats)
    assert any("n=40" in c and "standard error of kappa" in c for c in caveats)
    assert any("400-char" in c for c in caveats)
    assert not any("imported" in c or "held-out" in c for c in caveats)


async def test_calibrate_a_systematic_compression_lowers_kappa_and_can_fail(evals_session):
    judge = FakeJudge(grade_fn=lambda job: min(_index(job) % 4, 2))
    topic_id, _, run_id = await _labeled(evals_session, 40, judge)

    strict = await calibrate(evals_session, "topic", run_id, min_kappa=0.95)
    lenient = await calibrate(evals_session, "topic", run_id, min_kappa=0.5)

    assert 0.5 < strict.kappa < 0.95 and strict.kappa == pytest.approx(lenient.kappa)
    assert strict.confusion[3] == [0, 0, 10, 0]  # every human 3 came back as a 2
    assert strict.exact == 0.75
    assert not strict.passed and lenient.passed
    check = await calibration_for_run(evals_session, topic_id, run_id)
    assert check is not None and check.passed  # the newest check is the lenient one
    assert check.detail["threshold"] == 0.5


async def test_calibrate_leaves_out_the_fewshot_exemplars(evals_session):
    exemplar_idx = (0, 1, 2, 3)

    def grade(job):  # wrong only where the judge was shown the human answer
        i = _index(job)
        return (i + 1) % 4 if i in exemplar_idx else i % 4

    _, _, run_id = await _labeled(
        evals_session, 40, FakeJudge(grade_fn=grade), exemplar_idx=exemplar_idx
    )

    result = await calibrate(evals_session, "topic", run_id)

    assert (result.n, result.n_excluded_exemplars) == (36, 4)
    assert result.kappa == pytest.approx(1.0)


async def test_calibrate_uses_the_held_out_half_only_when_asked(evals_session):
    topic_id, job_ids = await _seed(evals_session, 40)
    by_title = {f"job-{i:03d}": job_id for i, job_id in enumerate(job_ids)}
    dev, held_out = split_dev_test(sorted(job_ids, key=str), seed=0)
    dev_titles = {title for title, job_id in by_title.items() if job_id in set(dev)}

    def grade(job):  # wrong on every dev job, right on every held-out job
        return (_index(job) + 1) % 4 if job.title in dev_titles else _index(job) % 4

    run_id = await label_topic(evals_session, topic_id, FakeJudge(grade_fn=grade), [], backoff=0)

    split = await calibrate(evals_session, "topic", run_id, use_split=True)
    everything = await calibrate(evals_session, "topic", run_id)

    assert (split.split, split.n) == ("held-out half", len(held_out))
    assert split.kappa == pytest.approx(1.0)
    assert (everything.split, everything.n) == ("all non-exemplar", 40)
    assert everything.exact == 0.5 and everything.kappa < split.kappa


async def test_calibrate_needs_enough_pairs(evals_session):
    _, _, run_id = await _labeled(evals_session, 8, FakeJudge(grade_fn=_index_grade))

    with pytest.raises(ValueError, match="at least"):
        await calibrate(evals_session, "topic", run_id)


async def test_calibrate_rejects_a_run_of_another_topic(evals_session):
    _, _, run_id = await _labeled(evals_session, 12, FakeJudge(grade_fn=_index_grade))
    await _seed(evals_session, 12, name="other")

    with pytest.raises(LookupError):
        await calibrate(evals_session, "other", run_id)
    with pytest.raises(LookupError):
        await calibrate(evals_session, "topic", UUID(int=0))
