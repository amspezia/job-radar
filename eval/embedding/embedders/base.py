"""What a candidate embedder is, independent of who serves it."""

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EmbedderSpec:
    """One row of `eval/embedders.toml` — the only place a candidate is defined."""

    name: str
    model: str
    backend: str = "ollama"
    num_ctx: int = 8192
    doc_prefix: str = ""
    hyde_prefix: str | None = None
    mrl: bool = False
    dim: int | None = None
    incumbent: bool = False

    def __post_init__(self) -> None:
        if self.dim is not None and not self.mrl:
            raise ValueError(
                f"{self.name}: dim = {self.dim} requires mrl = true — slicing a vector that "
                "was not trained with Matryoshka loss is not a dimension reduction"
            )

    @property
    def effective_hyde_prefix(self) -> str:
        """Production embeds HyDE text as a *document*, so that is the default."""
        return self.doc_prefix if self.hyde_prefix is None else self.hyde_prefix


@dataclass
class EmbedResult:
    vector: np.ndarray  # float32, native dimension — `dim` is applied at ranking time
    n_tokens: int | None
    truncated: bool


class Embedder(Protocol):
    spec: EmbedderSpec

    async def ready(self) -> None: ...
    async def embed(self, text: str) -> EmbedResult: ...
    def describe(self) -> dict: ...


def apply_dim(vec: np.ndarray, dim: int | None) -> np.ndarray:
    """Truncate a Matryoshka vector to `dim` and re-normalize.

    Ollama's vectors are already unit-norm (F4), so full-dimension use needs no
    work — but a slice is not, and comparing un-normalized vectors by dot
    product is not cosine.
    """
    if dim is None:
        return vec
    if dim > vec.shape[0]:
        raise ValueError(f"cannot truncate a {vec.shape[0]}-dimension vector to {dim}")
    sliced = vec[:dim]
    norm = float(np.linalg.norm(sliced))
    if norm == 0.0:
        raise ValueError(f"the first {dim} components are all zero; cannot re-normalize")
    return sliced / norm


def fingerprint(backend: str, digest: str, num_ctx: int) -> str:
    """Cache key for "these exact weights, served this way": sha256 of `backend|digest|num_ctx`.

    The digest, not the tag, identifies the weights: re-pulling a moved tag
    invalidates the cached vectors, renaming a spec does not.
    """
    return hashlib.sha256(f"{backend}|{digest}|{num_ctx}".encode()).hexdigest()
