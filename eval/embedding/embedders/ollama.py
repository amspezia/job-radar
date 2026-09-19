"""The Ollama-served embedder."""

import numpy as np

from eval.embedding.embedders.base import EmbedderSpec, EmbedResult, fingerprint
from eval.llm.ollama import ContextOverflow, OllamaClient


class OllamaEmbedder:
    """Embeds one text per call, recording what Ollama actually did with it."""

    def __init__(self, spec: EmbedderSpec, client: OllamaClient) -> None:
        self.spec = spec
        self._client = client
        self._digest: str | None = None
        self._quantization: str | None = None
        self._runtime_version: str | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.spec.backend, self._require_digest(), self.spec.num_ctx)

    def _require_digest(self) -> str:
        if self._digest is None:
            raise RuntimeError(
                f"{self.spec.name}: call ready() first — the model digest is unknown"
            )
        return self._digest

    async def ready(self) -> None:
        """Confirm the tag is pulled, record what is serving it, and pay the load."""
        infos = await self._client.tags()
        info = infos.get(self.spec.model)
        if info is None:
            raise RuntimeError(
                f"{self.spec.name}: Ollama has no model {self.spec.model!r} "
                f"(has: {', '.join(sorted(infos)) or 'nothing'}) — "
                f"run: ollama pull {self.spec.model}"
            )
        self._digest = info.digest
        self._quantization = info.quantization
        self._runtime_version = await self._client.version()
        await self._client.warm_embedder(self.spec.model, self.spec.num_ctx)

    async def embed(self, text: str) -> EmbedResult:
        """Embed `text`, refusing truncation first so that truncation is *observed*.

        The retry is the only honest detector (R-M7): `prompt_eval_count` is
        recorded but never used for it, because nomic and bge-m3 saturate at
        2,048 tokens regardless of `num_ctx`.
        """
        truncated = False
        try:
            response = await self._client.embed(
                self.spec.model, text, num_ctx=self.spec.num_ctx, truncate=False
            )
        except ContextOverflow:
            response = await self._client.embed(
                self.spec.model, text, num_ctx=self.spec.num_ctx, truncate=True
            )
            truncated = True
        return EmbedResult(
            vector=np.asarray(response.vector, dtype=np.float32),
            n_tokens=response.prompt_eval_count,
            truncated=truncated,
        )

    def describe(self) -> dict:
        return {
            "model": self.spec.model,
            "digest": self._require_digest(),
            "quantization": self._quantization,
            "runtime_version": self._runtime_version,
            "num_ctx": self.spec.num_ctx,
            "fingerprint": self.fingerprint,
        }
