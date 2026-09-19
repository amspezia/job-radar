import asyncio
import hashlib
import logging
import re
from dataclasses import replace
from uuid import UUID

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.cache import VectorCache
from eval.embedding.embedders.base import EmbedderSpec, EmbedResult, fingerprint
from eval.embedding.methods.base import TopicView, load_topic_view
from eval.embedding.methods.dense import DenseMethod
from eval.embedding.models import (
    EmbeddingEmbedder,
    EmbeddingJob,
    EmbeddingTopic,
    EmbeddingTopicJob,
    EmbeddingVector,
)

SPEC = EmbedderSpec(
    name="fake-v1",
    model="fake-embed",
    num_ctx=8192,
    doc_prefix="search_document: ",
    mrl=True,
)
HYDE = ("hyde zero", "hyde one", "hyde two")
NATIVE_DIM = 16


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def native_vector(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    vector = np.random.default_rng(seed).standard_normal(NATIVE_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    """Hash-derived vectors, a call log, and switches for truncation, failure and latency."""

    def __init__(
        self,
        spec: EmbedderSpec,
        *,
        digest: str = "sha256:fake",
        truncated_texts: frozenset[str] = frozenset(),
        fail_after: int | None = None,
        yield_control: bool = False,
    ) -> None:
        self.spec = spec
        self.digest = digest
        self.truncated_texts = truncated_texts
        self.fail_after = fail_after
        self.yield_control = yield_control
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def ready(self) -> None:
        return None

    async def embed(self, text: str) -> EmbedResult:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("embedder down")
        self.calls.append(text)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.yield_control:
            await asyncio.sleep(0)
        self.in_flight -= 1
        return EmbedResult(
            vector=native_vector(text),
            n_tokens=len(text.split()),
            truncated=text in self.truncated_texts,
        )

    def describe(self) -> dict:
        return {
            "model": self.spec.model,
            "digest": self.digest,
            "quantization": "F16",
            "runtime_version": "0.0.0",
            "num_ctx": self.spec.num_ctx,
            "fingerprint": fingerprint(self.spec.backend, self.digest, self.spec.num_ctx),
        }


async def seed_topic(
    session: AsyncSession, embed_texts: list[str], hyde_texts: tuple[str, ...] = HYDE
) -> list[UUID]:
    topic = EmbeddingTopic(
        tier="A",
        name="topic-1",
        status="ready",
        profile_snapshot={},
        query_inputs={"hyde_texts": list(hyde_texts)},
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


def build(
    session: AsyncSession, spec: EmbedderSpec = SPEC, recipe: str = "hyde_mean", **embedder_kwargs
) -> tuple[FakeEmbedder, DenseMethod]:
    embedder = FakeEmbedder(spec, **embedder_kwargs)
    return embedder, DenseMethod(embedder, VectorCache(session), recipe)


def reference_ranking(
    view: TopicView, spec: EmbedderSpec, hyde_index: int | None = None
) -> list[tuple[UUID, float]]:
    """Brute force on sliced, re-normalized native vectors — no apply_dim, no cosine_rank."""

    def unit_slice(text: str) -> np.ndarray:
        vector = native_vector(text)[: spec.dim].astype(np.float64)
        return vector / np.sqrt(np.sum(vector * vector))

    docs = np.array([unit_slice(spec.doc_prefix + doc.embed_text) for doc in view.docs])
    hyde = np.array(
        [unit_slice(spec.effective_hyde_prefix + text) for text in view.query_inputs["hyde_texts"]]
    )
    query = hyde.mean(axis=0) if hyde_index is None else hyde[hyde_index]
    scores = (docs @ query) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))
    order = sorted(range(len(view.docs)), key=lambda i: (-scores[i], view.docs[i].id))
    return [(view.docs[i].id, float(scores[i])) for i in order]


async def cached_shas(session: AsyncSession) -> set[str]:
    return set((await session.execute(select(EmbeddingVector.text_sha))).scalars())


async def n_cached(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(EmbeddingVector))


async def test_sends_and_caches_the_exact_prefixed_text(evals_session: AsyncSession):
    await seed_topic(evals_session, ["alpha", "beta"])
    view = await load_topic_view(evals_session, "topic-1")
    embedder, method = build(evals_session)

    await method.prepare(view)

    sent = {"search_document: alpha", "search_document: beta"} | {
        f"search_document: {text}" for text in HYDE
    }
    assert set(embedder.calls) == sent
    assert await cached_shas(evals_session) == {sha(text) for text in sent}
    row = await evals_session.scalar(select(EmbeddingEmbedder))
    assert row.fingerprint == embedder.describe()["fingerprint"]


@pytest.mark.parametrize(
    ("hyde_prefix", "expected"),
    [(None, "search_document: "), ("search_query: ", "search_query: ")],
)
async def test_hyde_texts_go_through_the_cache_with_the_effective_prefix(
    evals_session: AsyncSession, hyde_prefix: str | None, expected: str
):
    await seed_topic(evals_session, ["alpha"])
    view = await load_topic_view(evals_session, "topic-1")
    spec = replace(SPEC, hyde_prefix=hyde_prefix)
    embedder, method = build(evals_session, spec)

    await method.prepare(view)

    sent_hyde = {expected + text for text in HYDE}
    assert sent_hyde <= set(embedder.calls)
    assert {sha(text) for text in sent_hyde} <= await cached_shas(evals_session)
    assert spec.effective_hyde_prefix == expected


async def test_changing_the_doc_prefix_is_a_new_cache_entry(evals_session: AsyncSession):
    await seed_topic(evals_session, ["alpha", "beta"])
    view = await load_topic_view(evals_session, "topic-1")
    first, first_method = build(evals_session)
    await first_method.prepare(view)
    cached = await n_cached(evals_session)

    other_spec = replace(SPEC, doc_prefix="passage: ")
    second, second_method = build(evals_session, other_spec)
    await second_method.prepare(view)

    assert first_method.fingerprint() != second_method.fingerprint()
    assert {"passage: alpha", "passage: beta"} <= set(second.calls)
    # Every text carries a different prefix, so nothing of the first run can be a hit.
    assert len(second.calls) == len(first.calls)
    assert await n_cached(evals_session) == 2 * cached


@pytest.mark.parametrize(
    ("spec", "recipe", "digest"),
    [
        (replace(SPEC, doc_prefix="passage: "), "hyde_mean", "sha256:fake"),
        (replace(SPEC, hyde_prefix="search_query: "), "hyde_mean", "sha256:fake"),
        (replace(SPEC, dim=8), "hyde_mean", "sha256:fake"),
        (SPEC, "hyde_text:0", "sha256:fake"),
        (SPEC, "hyde_mean", "sha256:other-weights"),
    ],
)
async def test_the_method_fingerprint_covers_everything_that_changes_the_ranking(
    evals_session: AsyncSession, spec: EmbedderSpec, recipe: str, digest: str
):
    _, baseline = build(evals_session)
    _, same = build(evals_session)
    _, changed = build(evals_session, spec, recipe, digest=digest)

    assert baseline.fingerprint() == same.fingerprint()
    assert baseline.fingerprint() != changed.fingerprint()


async def test_a_second_prepare_makes_zero_embed_calls(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(6))
    view = await load_topic_view(evals_session, "topic-1")
    embedder, method = build(evals_session)

    await method.prepare(view)
    first_ranking = method.rank(view)
    assert len(embedder.calls) == 6 + len(HYDE)
    assert method.stats()["docs_per_s"] > 0

    await method.prepare(view)  # same instance
    assert len(embedder.calls) == 6 + len(HYDE)
    assert method.stats()["docs_per_s"] is None  # nothing was embedded, so no throughput
    assert method.rank(view) == first_ranking

    fresh, fresh_method = build(evals_session)  # a new process, same cache
    await fresh_method.prepare(view)
    assert fresh.calls == []
    assert fresh_method.rank(view) == first_ranking


async def test_a_dim_variant_reuses_native_vectors_and_ranks_on_sliced_ones(
    evals_session: AsyncSession,
):
    await seed_topic(evals_session, doc_texts(20))
    view = await load_topic_view(evals_session, "topic-1")
    native, native_method = build(evals_session)
    await native_method.prepare(view)
    native_ranking = native_method.rank(view)

    spec = replace(SPEC, name="fake-v1-d8", dim=8)
    embedder, method = build(evals_session, spec)
    await method.prepare(view)
    ranking = method.rank(view)

    assert embedder.calls == []  # same weights, so same fingerprint, same cached vectors
    assert method.stats()["embedder_fingerprint"] == native_method.stats()["embedder_fingerprint"]
    expected = reference_ranking(view, spec)
    assert [doc_id for doc_id, _ in ranking] == [doc_id for doc_id, _ in expected]
    assert np.allclose([score for _, score in ranking], [score for _, score in expected])
    assert [score for _, score in ranking] != [score for _, score in native_ranking]
    assert len(native.calls) == 20 + len(HYDE)


async def test_hyde_mean_and_a_single_hyde_text_rank_differently(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(20))
    view = await load_topic_view(evals_session, "topic-1")
    embedder, mean_method = build(evals_session)
    await mean_method.prepare(view)
    calls_after_mean = len(embedder.calls)

    single_embedder, single_method = build(evals_session, recipe="hyde_text:0")
    await single_method.prepare(view)

    assert single_embedder.calls == []  # every hyde_text:<i> recipe is free once one has run
    assert calls_after_mean == 20 + len(HYDE)
    mean_ranking, single_ranking = mean_method.rank(view), single_method.rank(view)
    assert [doc_id for doc_id, _ in mean_ranking] != [doc_id for doc_id, _ in single_ranking]
    assert [doc_id for doc_id, _ in single_ranking] == [
        doc_id for doc_id, _ in reference_ranking(view, SPEC, hyde_index=0)
    ]
    assert [doc_id for doc_id, _ in mean_ranking] == [
        doc_id for doc_id, _ in reference_ranking(view, SPEC)
    ]
    assert mean_method.name == "fake-v1"
    assert single_method.name == "fake-v1:hyde_text:0"


async def test_recipes_are_validated(evals_session: AsyncSession):
    with pytest.raises(ValueError, match="unknown query recipe"):
        build(evals_session, recipe="hyde_text:x")
    with pytest.raises(ValueError, match="unknown query recipe"):
        build(evals_session, recipe="mean")

    await seed_topic(evals_session, doc_texts(3))
    view = await load_topic_view(evals_session, "topic-1")
    _, method = build(evals_session, recipe=f"hyde_text:{len(HYDE)}")
    await method.prepare(view)
    with pytest.raises(IndexError, match="needs HyDE text 3"):
        method.rank(view)


async def test_a_topic_without_hyde_texts_cannot_be_prepared(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(2), hyde_texts=())
    view = await load_topic_view(evals_session, "topic-1")
    embedder, method = build(evals_session)

    with pytest.raises(ValueError, match="hyde_texts"):
        await method.prepare(view)
    assert embedder.calls == []


async def test_truncated_flags_are_counted_over_docs_hyde_new_and_cached(
    evals_session: AsyncSession,
):
    await seed_topic(evals_session, ["alpha", "beta", "gamma"])
    view = await load_topic_view(evals_session, "topic-1")
    flagged = frozenset(
        {"search_document: alpha", "search_document: gamma", "search_document: hyde one"}
    )
    _, method = build(evals_session, truncated_texts=flagged)

    await method.prepare(view)
    assert method.stats()["n_truncated"] == 3  # two docs and one HyDE text

    _, cached_method = build(evals_session)  # cached flags count, whatever the new embedder says
    await cached_method.prepare(view)
    assert cached_method.stats()["n_truncated"] == 3


async def test_identical_texts_are_embedded_once(evals_session: AsyncSession):
    ids = await seed_topic(evals_session, ["repost", "repost", "other"])
    view = await load_topic_view(evals_session, "topic-1")
    embedder, method = build(evals_session)

    await method.prepare(view)

    assert embedder.calls.count("search_document: repost") == 1
    assert len(embedder.calls) == 2 + len(HYDE)
    ranking = method.rank(view)
    order = [doc_id for doc_id, _ in ranking]
    scores = dict(ranking)
    assert scores[ids[0]] == scores[ids[1]]
    assert order.index(ids[0]) < order.index(ids[1])  # ties fall back to ascending id


async def test_partial_progress_survives_a_failure_and_the_rerun_embeds_the_rest(
    evals_session: AsyncSession,
):
    await seed_topic(evals_session, doc_texts(12))
    view = await load_topic_view(evals_session, "topic-1")
    total = 12 + len(HYDE)
    failing, failing_method = build(evals_session, fail_after=5)

    with pytest.raises(RuntimeError, match="embedder down"):
        await failing_method.prepare(view)

    assert len(failing.calls) == 5
    assert await n_cached(evals_session) == 5
    rerun, method = build(evals_session)
    await method.prepare(view)
    assert len(rerun.calls) == total - 5
    assert not set(rerun.calls) & set(failing.calls)
    assert await n_cached(evals_session) == total
    assert len(method.rank(view)) == 12


async def test_vectors_are_persisted_in_batches_of_a_hundred(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(230), hyde_texts=("only hyde",))
    view = await load_topic_view(evals_session, "topic-1")
    cache = VectorCache(evals_session)
    method = DenseMethod(FakeEmbedder(SPEC), cache)
    batch_sizes: list[int] = []
    put_many = cache.put_many

    async def spy(fingerprint: str, items: list) -> None:
        batch_sizes.append(len(items))
        await put_many(fingerprint, items)

    cache.put_many = spy
    await method.prepare(view)

    assert batch_sizes == [100, 100, 31]
    assert await n_cached(evals_session) == 231


async def test_progress_is_logged_every_25_texts(
    evals_session: AsyncSession, caplog: pytest.LogCaptureFixture
):
    await seed_topic(evals_session, doc_texts(57))
    view = await load_topic_view(evals_session, "topic-1")
    _, method = build(evals_session)

    with caplog.at_level(logging.INFO, logger="eval.embedding.methods.dense"):
        await method.prepare(view)

    progress = [
        m.groups()
        for r in caplog.records
        if (m := re.search(r"embedded (\d+)/(\d+) texts", r.message))
    ]
    assert progress == [("25", "60"), ("50", "60"), ("60", "60")]


@pytest.mark.parametrize("concurrency", [8, 2])
async def test_embed_calls_are_bounded_by_the_concurrency(
    evals_session: AsyncSession, concurrency: int
):
    await seed_topic(evals_session, doc_texts(20))
    view = await load_topic_view(evals_session, "topic-1")
    embedder = FakeEmbedder(SPEC, yield_control=True)
    method = DenseMethod(embedder, VectorCache(evals_session), concurrency=concurrency)

    await method.prepare(view)

    assert embedder.max_in_flight == concurrency


async def test_rank_needs_a_prepare_for_the_same_topic(evals_session: AsyncSession):
    await seed_topic(evals_session, doc_texts(3))
    view = await load_topic_view(evals_session, "topic-1")
    _, method = build(evals_session)

    with pytest.raises(RuntimeError, match=r"prepare\(\) must run before rank\(\)"):
        method.rank(view)

    await method.prepare(view)
    other = replace(view, topic_id=UUID(int=999), name="other")
    with pytest.raises(RuntimeError, match="prepared for another topic"):
        method.rank(other)
