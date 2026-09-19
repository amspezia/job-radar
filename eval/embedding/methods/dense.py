"""Dense retrieval: embed the corpus, embed the HyDE texts, rank by cosine."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict
from uuid import UUID

import numpy as np

from eval.embedding.cache import VectorCache
from eval.embedding.embedders.base import Embedder, apply_dim
from eval.embedding.methods.base import TopicView
from eval.embedding.ranking import cosine_rank

logger = logging.getLogger(__name__)

# Bumped when the meaning of a recipe changes, so old runs never compare as equal.
RECIPE_VERSION = "1"

_HYDE_MEAN = "hyde_mean"
_HYDE_TEXT = "hyde_text:"
_PROGRESS_LOG_EVERY = 25
# Small enough that a crash loses seconds of work, large enough to amortize the commit.
_PUT_BATCH = 100


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _parse_recipe(recipe: str) -> int | None:
    """Return the HyDE text index a recipe selects, or None for the mean of all of them."""
    if recipe == _HYDE_MEAN:
        return None
    if recipe.startswith(_HYDE_TEXT):
        index = recipe.removeprefix(_HYDE_TEXT)
        if index.isdigit():
            return int(index)
    raise ValueError(f"unknown query recipe {recipe!r}; expected {_HYDE_MEAN!r} or 'hyde_text:<i>'")


class DenseMethod:
    """One embedder, one query recipe, over a topic's frozen corpus.

    Both the corpus and the HyDE texts go through the same cache, so the second run of any
    variant — and every `hyde_text:<i>` recipe of a variant already run — costs no embedding.
    """

    def __init__(
        self,
        embedder: Embedder,
        cache: VectorCache,
        query_recipe: str = _HYDE_MEAN,
        concurrency: int = 8,
    ) -> None:
        self._hyde_index = _parse_recipe(query_recipe)
        self._embedder = embedder
        self._cache = cache
        self._recipe = query_recipe
        self._concurrency = concurrency
        spec = embedder.spec
        self.name = spec.name if query_recipe == _HYDE_MEAN else f"{spec.name}:{query_recipe}"
        self._prepared_for: UUID | None = None
        self._doc_matrix: np.ndarray | None = None
        self._hyde_matrix: np.ndarray | None = None
        self._docs_per_s: float | None = None
        self._n_truncated = 0

    def fingerprint(self) -> str:
        """Everything that changes the ranking: the weights, the text sent, the query."""
        spec = self._embedder.spec
        payload = {
            "embedder": self._embedder.describe()["fingerprint"],
            "doc_prefix": spec.doc_prefix,
            "hyde_prefix": spec.effective_hyde_prefix,
            "dim": spec.dim,
            "query_recipe": self._recipe,
            "recipe_version": RECIPE_VERSION,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def describe(self) -> dict:
        spec = self._embedder.spec
        return {
            "kind": "dense",
            "name": self.name,
            "query_recipe": self._recipe,
            "recipe_version": RECIPE_VERSION,
            "spec": asdict(spec),
            "effective_hyde_prefix": spec.effective_hyde_prefix,
            "embedder": self._embedder.describe(),
        }

    async def prepare(self, topic: TopicView) -> None:
        """Fill the cache with everything this run needs, then hold the matrices in memory."""
        spec = self._embedder.spec
        hyde_inputs = topic.query_inputs.get("hyde_texts") or []
        if not hyde_inputs:
            raise ValueError(f"topic {topic.name!r} has no query_inputs['hyde_texts'] to embed")

        doc_texts = [spec.doc_prefix + doc.embed_text for doc in topic.docs]
        hyde_texts = [spec.effective_hyde_prefix + text for text in hyde_inputs]
        doc_shas = [_sha(text) for text in doc_texts]
        hyde_shas = [_sha(text) for text in hyde_texts]

        # Identical texts (reposts, and a repeated HyDE text) are one embed call and one row.
        wanted = dict(zip(doc_shas + hyde_shas, doc_texts + hyde_texts, strict=True))
        describe = self._embedder.describe()
        fingerprint = describe["fingerprint"]
        await self._cache.ensure_embedder_row(fingerprint, spec, describe)
        vectors = await self._cache.get_many(fingerprint, list(wanted))
        missing = [(sha, text) for sha, text in wanted.items() if sha not in vectors]
        # Throughput describes this call only: a fully cached re-run must not inherit the last.
        self._docs_per_s = None
        if missing:
            await self._embed_missing(fingerprint, missing, vectors, set(doc_shas))

        self._n_truncated = sum(1 for _, _, truncated in vectors.values() if truncated)
        self._doc_matrix = np.vstack([vectors[sha][0] for sha in doc_shas])
        self._hyde_matrix = np.vstack([vectors[sha][0] for sha in hyde_shas])
        self._prepared_for = topic.topic_id

    async def _embed_missing(
        self,
        fingerprint: str,
        missing: list[tuple[str, str]],
        vectors: dict[str, tuple[np.ndarray, int | None, bool]],
        doc_shas: set[str],
    ) -> None:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def embed(sha: str, text: str) -> tuple[str, np.ndarray, int | None, bool]:
            async with semaphore:
                result = await self._embedder.embed(text)
            return sha, result.vector, result.n_tokens, result.truncated

        total = len(missing)
        logger.info("embedder=%s embedding %d uncached texts", self.name, total)
        tasks = [asyncio.create_task(embed(sha, text)) for sha, text in missing]
        batch: list[tuple[str, np.ndarray, int | None, bool]] = []
        started = time.perf_counter()
        done = 0
        try:
            for completed in asyncio.as_completed(tasks):
                sha, vector, n_tokens, truncated = await completed
                vectors[sha] = (vector, n_tokens, truncated)
                batch.append((sha, vector, n_tokens, truncated))
                if len(batch) >= _PUT_BATCH:
                    await self._cache.put_many(fingerprint, batch)
                    batch.clear()
                done += 1
                if done % _PROGRESS_LOG_EVERY == 0 or done == total:
                    logger.info("embedder=%s embedded %d/%d texts", self.name, done, total)
        finally:
            # Also the failure path: whatever finished is banked before the error propagates.
            for task in tasks:
                task.cancel()
            for outcome in await asyncio.gather(*tasks, return_exceptions=True):
                # Finished after the task that failed, so the loop above never saw them.
                if isinstance(outcome, tuple) and outcome[0] not in vectors:
                    vectors[outcome[0]] = outcome[1:]
                    batch.append(outcome)
            if batch:
                await self._cache.put_many(fingerprint, batch)
        elapsed = time.perf_counter() - started
        new_docs = sum(1 for sha, _ in missing if sha in doc_shas)
        self._docs_per_s = new_docs / elapsed if new_docs and elapsed > 0 else None

    def rank(self, topic: TopicView) -> list[tuple[UUID, float]]:
        if self._doc_matrix is None or self._hyde_matrix is None:
            raise RuntimeError(f"{self.name}: prepare() must run before rank()")
        if self._prepared_for != topic.topic_id:
            raise RuntimeError(f"{self.name}: prepared for another topic than {topic.name!r}")

        dim = self._embedder.spec.dim
        matrix = np.vstack([apply_dim(vector, dim) for vector in self._doc_matrix])
        hyde = [apply_dim(vector, dim) for vector in self._hyde_matrix]
        if self._hyde_index is None:
            # A mean of unit vectors is not unit-norm; cosine does not care.
            query = np.mean(hyde, axis=0)
        elif self._hyde_index >= len(hyde):
            raise IndexError(
                f"recipe {self._recipe!r} needs HyDE text {self._hyde_index}, "
                f"but topic {topic.name!r} has {len(hyde)}"
            )
        else:
            query = hyde[self._hyde_index]
        return cosine_rank(matrix, [doc.id for doc in topic.docs], query)

    def stats(self) -> dict:
        return {
            "docs_per_s": self._docs_per_s,
            "n_truncated": self._n_truncated,
            "embedder_fingerprint": self._embedder.describe()["fingerprint"],
        }
