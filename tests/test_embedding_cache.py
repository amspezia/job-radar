import hashlib

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.cache import VectorCache
from eval.embedding.embedders.base import EmbedderSpec, fingerprint
from eval.embedding.models import EmbeddingEmbedder, EmbeddingVector

SPEC = EmbedderSpec(
    name="fake-v1",
    model="fake-embed",
    num_ctx=8192,
    doc_prefix="search_document: ",
    mrl=True,
)
FINGERPRINT = fingerprint(SPEC.backend, "sha256:aaa", SPEC.num_ctx)
DESCRIBE = {
    "model": SPEC.model,
    "digest": "sha256:aaa",
    "quantization": "F16",
    "runtime_version": "0.30.11",
    "num_ctx": SPEC.num_ctx,
    "fingerprint": FINGERPRINT,
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def unit(dim: int, seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


async def test_ensure_embedder_row_is_written_once(evals_session: AsyncSession):
    cache = VectorCache(evals_session)
    await cache.ensure_embedder_row(FINGERPRINT, SPEC, DESCRIBE)
    await cache.ensure_embedder_row(FINGERPRINT, SPEC, DESCRIBE | {"quantization": "Q8_0"})

    rows = (await evals_session.execute(select(EmbeddingEmbedder))).scalars().all()
    assert len(rows) == 1
    assert rows[0].digest == "sha256:aaa"
    assert rows[0].quantization == "F16"  # the first write wins
    assert rows[0].config == {
        "name": "fake-v1",
        "model": "fake-embed",
        "backend": "ollama",
        "num_ctx": 8192,
        "doc_prefix": "search_document: ",
        "hyde_prefix": None,
        "mrl": True,
        "dim": None,
        "incumbent": False,
    }


async def test_vectors_round_trip_at_any_dimension(evals_session: AsyncSession):
    cache = VectorCache(evals_session)
    await cache.ensure_embedder_row(FINGERPRINT, SPEC, DESCRIBE)
    small, large = unit(3, 1), unit(1024, 2)
    await cache.put_many(
        FINGERPRINT,
        [(sha("small"), small, 7, False), (sha("large"), large, None, True)],
    )

    found = await cache.get_many(FINGERPRINT, [sha("small"), sha("large"), sha("absent")])

    assert set(found) == {sha("small"), sha("large")}
    for cached, original in ((found[sha("small")][0], small), (found[sha("large")][0], large)):
        assert cached.dtype == np.float32
        assert np.allclose(cached, original)
    assert found[sha("small")][1:] == (7, False)
    assert found[sha("large")][1:] == (None, True)


async def test_put_many_keeps_the_cached_vector(evals_session: AsyncSession):
    cache = VectorCache(evals_session)
    await cache.ensure_embedder_row(FINGERPRINT, SPEC, DESCRIBE)
    first = unit(4, 3)
    await cache.put_many(FINGERPRINT, [(sha("t"), first, 11, False)])
    await cache.put_many(FINGERPRINT, [(sha("t"), unit(4, 4), 99, True)])

    vector, n_tokens, truncated = (await cache.get_many(FINGERPRINT, [sha("t")]))[sha("t")]
    assert np.allclose(vector, first)
    assert (n_tokens, truncated) == (11, False)


async def test_a_cache_entry_belongs_to_one_fingerprint(evals_session: AsyncSession):
    cache = VectorCache(evals_session)
    other = fingerprint(SPEC.backend, "sha256:bbb", SPEC.num_ctx)
    await cache.ensure_embedder_row(FINGERPRINT, SPEC, DESCRIBE)
    await cache.ensure_embedder_row(other, SPEC, DESCRIBE | {"digest": "sha256:bbb"})
    await cache.put_many(FINGERPRINT, [(sha("t"), unit(4, 5), None, False)])

    assert await cache.get_many(other, [sha("t")]) == {}


async def test_reads_and_writes_chunk_beyond_one_statement(evals_session: AsyncSession):
    cache = VectorCache(evals_session)
    await cache.ensure_embedder_row(FINGERPRINT, SPEC, DESCRIBE)
    shas = [sha(f"text-{i}") for i in range(1200)]
    await cache.put_many(FINGERPRINT, [(s, unit(3, i), i, False) for i, s in enumerate(shas)])

    stored = await evals_session.scalar(select(func.count()).select_from(EmbeddingVector))
    found = await cache.get_many(FINGERPRINT, [*shas, sha("absent")])

    assert stored == 1200
    assert len(found) == 1200
    assert np.allclose(found[shas[1199]][0], unit(3, 1199))
