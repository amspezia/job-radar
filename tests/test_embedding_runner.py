import hashlib
from uuid import UUID, uuid4

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.cache import VectorCache
from eval.embedding.embedders.base import EmbedderSpec, EmbedResult, fingerprint
from eval.embedding.methods.base import TopicView, load_topic_view
from eval.embedding.methods.dense import DenseMethod
from eval.embedding.models import (
    EmbeddingEmbedder,
    EmbeddingJob,
    EmbeddingRun,
    EmbeddingRunRanking,
    EmbeddingTopic,
    EmbeddingTopicJob,
    EmbeddingVector,
)
from eval.embedding.runner import run_method

SPEC = EmbedderSpec(name="fake-v1", model="fake-embed", doc_prefix="search_document: ")
DIGEST = "sha256:fake"
HYDE = ("hyde zero", "hyde one", "hyde two")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FakeEmbedder:
    def __init__(
        self, truncated_texts: frozenset[str] = frozenset(), fail_after: int | None = None
    ) -> None:
        self.spec = SPEC
        self.truncated_texts = truncated_texts
        self.fail_after = fail_after
        self.calls: list[str] = []

    async def ready(self) -> None:
        return None

    async def embed(self, text: str) -> EmbedResult:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("embedder down")
        self.calls.append(text)
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
        return EmbedResult(
            vector=vector / np.linalg.norm(vector),
            n_tokens=len(text.split()),
            truncated=text in self.truncated_texts,
        )

    def describe(self) -> dict:
        return {
            "model": SPEC.model,
            "digest": DIGEST,
            "quantization": "F16",
            "runtime_version": "0.0.0",
            "num_ctx": SPEC.num_ctx,
            "fingerprint": fingerprint(SPEC.backend, DIGEST, SPEC.num_ctx),
        }


class FakeMethod:
    """A method that is not dense: no embedder, `stats()` is empty."""

    name = "fake-method"

    def __init__(
        self,
        session: AsyncSession | None = None,
        *,
        fail_in_prepare: bool = False,
        ranking: list[tuple[UUID, float]] | None = None,
    ) -> None:
        self._session = session
        self._fail_in_prepare = fail_in_prepare
        self._ranking = ranking
        self.events: list[str] = []
        self.status_during_prepare: str | None = None

    def fingerprint(self) -> str:
        return "fake-fingerprint"

    def describe(self) -> dict:
        return {"kind": "fake"}

    async def prepare(self, topic: TopicView) -> None:
        self.events.append("prepare")
        if self._session is not None:
            self.status_during_prepare = await self._session.scalar(select(EmbeddingRun.status))
        if self._fail_in_prepare:
            raise RuntimeError("boom")

    def rank(self, topic: TopicView) -> list[tuple[UUID, float]]:
        self.events.append("rank")
        if self._ranking is not None:
            return self._ranking
        # Best first is the reverse of the corpus order, so rank 1 is not the first row seeded.
        docs = list(reversed(topic.docs))
        return [(doc.id, float(len(docs) - i)) for i, doc in enumerate(docs)]

    def stats(self) -> dict:
        return {}


async def seed_topic(session: AsyncSession, embed_texts: list[str]) -> list[UUID]:
    topic = EmbeddingTopic(
        tier="A",
        name="topic-1",
        status="ready",
        profile_snapshot={},
        query_inputs={"hyde_texts": list(HYDE)},
        builder="test",
    )
    jobs = [
        EmbeddingJob(
            id=UUID(int=i + 1),
            origin="test",
            origin_id=f"job-{i}",
            source="test",
            title=f"Job {i}",
            company="Acme",
            url=f"https://example.test/{i}",
            description=f"Description {i}",
            embed_text=text,
            embed_text_sha=sha(text),
        )
        for i, text in enumerate(embed_texts)
    ]
    session.add_all([topic, *jobs])
    await session.flush()
    session.add_all([EmbeddingTopicJob(topic_id=topic.id, job_id=job.id) for job in jobs])
    await session.commit()
    return [job.id for job in jobs]


def doc_texts(n: int) -> list[str]:
    return [f"job posting number {i} about python and postgres" for i in range(n)]


async def fetch_run(session: AsyncSession, run_id: UUID) -> EmbeddingRun:
    # UPDATEs bypass the identity map, so re-read the row instead of trusting the added object.
    return (
        await session.execute(
            select(EmbeddingRun)
            .where(EmbeddingRun.id == run_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def stored_ranking(session: AsyncSession, run_id: UUID) -> list[tuple[int, UUID, float]]:
    rows = await session.execute(
        select(EmbeddingRunRanking.rank, EmbeddingRunRanking.job_id, EmbeddingRunRanking.score)
        .where(EmbeddingRunRanking.run_id == run_id)
        .order_by(EmbeddingRunRanking.rank)
    )
    return [tuple(row) for row in rows]


async def test_a_dense_run_stores_the_full_ranking_and_closes_the_run(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(12))
    flagged = frozenset({"search_document: " + doc_texts(12)[3]})
    embedder = FakeEmbedder(truncated_texts=flagged)
    method = DenseMethod(embedder, VectorCache(evals_session))

    run_id = await run_method(evals_session, "topic-1", method, git_sha="abc123")

    view = await load_topic_view(evals_session, "topic-1")
    expected = method.rank(view)
    stored = await stored_ranking(evals_session, run_id)
    assert [rank for rank, _, _ in stored] == list(range(1, 13))
    assert [(job_id, score) for _, job_id, score in stored] == expected
    run = await fetch_run(evals_session, run_id)
    assert run.status == "completed"
    assert run.name == "fake-v1"
    assert run.git_sha == "abc123"
    assert run.method_fingerprint == method.fingerprint()
    assert run.method_config["kind"] == "dense"
    assert run.embedder_fingerprint == embedder.describe()["fingerprint"]
    assert run.docs_per_s > 0
    assert run.n_truncated == 1
    assert run.seconds > 0
    assert run.finished_at is not None
    assert run.error is None
    embedder_row = await evals_session.get(EmbeddingEmbedder, run.embedder_fingerprint)
    assert embedder_row is not None


async def test_score_ties_are_ranked_by_ascending_job_id(evals_session: AsyncSession):
    ids = await seed_topic(evals_session, ["same text"] * 5)  # one vector, five documents
    method = DenseMethod(FakeEmbedder(), VectorCache(evals_session))

    run_id = await run_method(evals_session, "topic-1", method)

    stored = await stored_ranking(evals_session, run_id)
    assert [job_id for _, job_id, _ in stored] == sorted(ids)
    assert [rank for rank, _, _ in stored] == [1, 2, 3, 4, 5]
    assert len({score for _, _, score in stored}) == 1


async def test_a_method_that_is_not_dense_goes_through_unchanged(evals_session: AsyncSession):
    ids = await seed_topic(evals_session, doc_texts(4))
    method = FakeMethod(evals_session)

    run_id = await run_method(evals_session, "topic-1", method, git_sha="def456")

    assert method.events == ["prepare", "rank"]
    assert method.status_during_prepare == "running"  # visible before any work starts
    stored = await stored_ranking(evals_session, run_id)
    assert [(rank, job_id) for rank, job_id, _ in stored] == list(
        zip(range(1, 5), reversed(ids), strict=True)
    )
    run = await fetch_run(evals_session, run_id)
    assert run.status == "completed"
    assert run.name == "fake-method"
    assert run.method_fingerprint == "fake-fingerprint"
    assert run.method_config == {"kind": "fake"}
    assert run.git_sha == "def456"
    assert run.embedder_fingerprint is None
    assert run.docs_per_s is None
    assert run.n_truncated is None
    assert run.seconds is not None
    assert run.finished_at is not None


async def test_a_ranking_longer_than_one_insert_chunk_is_stored_whole(evals_session: AsyncSession):
    ids = await seed_topic(evals_session, [f"doc {i}" for i in range(1201)])

    run_id = await run_method(evals_session, "topic-1", FakeMethod())

    n, low, high, distinct = (
        await evals_session.execute(
            select(
                func.count(),
                func.min(EmbeddingRunRanking.rank),
                func.max(EmbeddingRunRanking.rank),
                func.count(func.distinct(EmbeddingRunRanking.rank)),
            ).where(EmbeddingRunRanking.run_id == run_id)
        )
    ).one()
    assert (n, low, high, distinct) == (1201, 1, 1201, 1201)
    assert (await stored_ranking(evals_session, run_id))[0][1] == ids[-1]


async def test_a_failing_prepare_marks_the_run_failed_and_reraises(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(3))

    with pytest.raises(RuntimeError, match="boom"):
        await run_method(evals_session, "topic-1", FakeMethod(fail_in_prepare=True))

    run = (await evals_session.execute(select(EmbeddingRun))).scalar_one()
    assert run.status == "failed"
    assert "boom" in run.error
    assert run.seconds is not None
    assert run.finished_at is not None
    assert run.embedder_fingerprint is None
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingRunRanking)) == 0


async def test_a_failed_ranking_insert_rolls_back_and_leaves_no_partial_ranking(
    evals_session: AsyncSession,
):
    ids = await seed_topic(evals_session, doc_texts(3))
    ranking = [(ids[0], 1.0), (uuid4(), 0.5)]  # the second is not a job: the FK rejects it
    method = FakeMethod(ranking=ranking)

    with pytest.raises(IntegrityError):
        await run_method(evals_session, "topic-1", method)

    run = (await evals_session.execute(select(EmbeddingRun))).scalar_one()
    assert run.status == "failed"
    assert run.error
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingRunRanking)) == 0


async def test_a_dense_run_that_dies_keeps_its_vectors_and_the_rerun_completes(
    evals_session: AsyncSession,
):
    await seed_topic(evals_session, doc_texts(10))
    failing = FakeEmbedder(fail_after=4)

    with pytest.raises(RuntimeError, match="embedder down"):
        await run_method(evals_session, "topic-1", DenseMethod(failing, VectorCache(evals_session)))

    failed = (await evals_session.execute(select(EmbeddingRun))).scalar_one()
    assert failed.status == "failed"
    assert "embedder down" in failed.error
    assert failed.embedder_fingerprint is None
    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingVector)) == 4

    healthy = FakeEmbedder()
    run_id = await run_method(
        evals_session, "topic-1", DenseMethod(healthy, VectorCache(evals_session))
    )

    assert len(healthy.calls) == 10 + len(HYDE) - 4
    run = await fetch_run(evals_session, run_id)
    assert run.status == "completed"
    assert run.embedder_fingerprint == healthy.describe()["fingerprint"]
    assert len(await stored_ranking(evals_session, run_id)) == 10


async def test_an_unknown_topic_creates_no_run(evals_session: AsyncSession):
    with pytest.raises(LookupError, match="no-such-topic"):
        await run_method(evals_session, "no-such-topic", FakeMethod())

    assert await evals_session.scalar(select(func.count()).select_from(EmbeddingRun)) == 0
