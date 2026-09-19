"""The embedding cache: one row per (embedder fingerprint, exact text sent)."""

from collections.abc import Sequence
from dataclasses import asdict
from itertools import batched

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from eval.embedding.embedders.base import EmbedderSpec
from eval.embedding.models import EmbeddingEmbedder, EmbeddingVector

# Postgres takes a large IN list, but a bounded one keeps the statement (and its plan) sane.
_SELECT_CHUNK = 1000
# Each row carries a full vector, so writes stay small enough to commit often.
_INSERT_CHUNK = 200


class VectorCache:
    """Stores vectors at their **native** dimension, keyed by the exact string sent.

    Native storage is what lets a Matryoshka `dim` variant re-use the vectors of the spec it
    is derived from: `dim` is applied at ranking time, never before the cache. The key is the
    sha of the prefixed text, so changing a prefix is a different cache entry, not a silent hit.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ensure_embedder_row(
        self, fingerprint: str, spec: EmbedderSpec, describe: dict
    ) -> None:
        """Record what serves this fingerprint; vectors and runs reference it."""
        await self._session.execute(
            pg_insert(EmbeddingEmbedder)
            .values(
                fingerprint=fingerprint,
                name=spec.name,
                model=spec.model,
                digest=describe["digest"],
                quantization=describe.get("quantization"),
                runtime_version=describe.get("runtime_version"),
                config=asdict(spec),
            )
            .on_conflict_do_nothing(index_elements=["fingerprint"])
        )
        await self._session.commit()

    async def get_many(
        self, fingerprint: str, shas: Sequence[str]
    ) -> dict[str, tuple[np.ndarray, int | None, bool]]:
        """Return the cached `(vector, n_tokens, truncated)` of every sha that has one."""
        found: dict[str, tuple[np.ndarray, int | None, bool]] = {}
        for chunk in batched(shas, _SELECT_CHUNK):
            rows = await self._session.execute(
                select(
                    EmbeddingVector.text_sha,
                    EmbeddingVector.vector,
                    EmbeddingVector.n_tokens,
                    EmbeddingVector.truncated,
                ).where(
                    EmbeddingVector.fingerprint == fingerprint,
                    EmbeddingVector.text_sha.in_(chunk),
                )
            )
            for text_sha, vector, n_tokens, truncated in rows:
                found[text_sha] = (vector, n_tokens, truncated)
        return found

    async def put_many(
        self, fingerprint: str, items: Sequence[tuple[str, np.ndarray, int | None, bool]]
    ) -> None:
        """Insert new vectors, keeping whatever is already cached.

        Commits per chunk: an embed run that dies halfway leaves its finished work behind and
        resumes as a cache hit.
        """
        for chunk in batched(items, _INSERT_CHUNK):
            await self._session.execute(
                pg_insert(EmbeddingVector)
                .values(
                    [
                        {
                            "fingerprint": fingerprint,
                            "text_sha": text_sha,
                            "vector": vector,
                            "n_tokens": n_tokens,
                            "truncated": truncated,
                        }
                        for text_sha, vector, n_tokens, truncated in chunk
                    ]
                )
                .on_conflict_do_nothing(index_elements=["fingerprint", "text_sha"])
            )
            await self._session.commit()
