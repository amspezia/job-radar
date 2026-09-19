"""Async Ollama client for the eval harness.

Deliberately not `job_radar.adapters.providers`: importing that pulls in prod
settings, the prod engine and Langfuse (F12/F13), and it hardcodes one model and
one `num_ctx`. The harness talks to several models, needs `num_ctx` and
`truncate` per call, and must import nothing from `job_radar`.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BASE_BACKOFF = 0.5  # seconds; doubles each retry


def _timeout(read: float) -> httpx.Timeout:
    """Split the read leg from the rest, as the prod provider does.

    Against a local Ollama, connect/write/pool are near-instant, so a flat
    timeout means one stuck request blocks for the whole read budget before it
    can be retried. Only the read leg legitimately takes a while.
    """
    return httpx.Timeout(connect=10.0, read=read, write=10.0, pool=10.0)


_META_TIMEOUT = _timeout(30.0)
_EMBED_TIMEOUT = _timeout(60.0)
_CHAT_TIMEOUT = _timeout(420.0)
# A cold model load (Gemma3:12b is ~8 GB) is far slower than any request it serves.
_WARM_TIMEOUT = _timeout(600.0)

# Ollama 0.30.11's body for an over-long input when truncation is refused.
_OVERFLOW_MARKER = "the input length exceeds the context length"


class ContextOverflow(Exception):
    """A `truncate=false` embed was refused: the input exceeds the context.

    The only reliable truncation detector. `prompt_eval_count` is not one:
    nomic-embed-text and bge-m3 saturate at 2,048 tokens whatever `num_ctx`
    says, so a count-based check reports zero truncated documents (R-M7).
    """


@dataclass
class EmbedResponse:
    vector: list[float]
    prompt_eval_count: int | None


@dataclass
class ChatResult:
    data: dict
    prompt_eval_count: int | None
    eval_count: int | None
    prompt_eval_seconds: float | None
    seconds: float


@dataclass
class TagInfo:
    digest: str
    quantization: str | None
    context_length: int | None


def _is_transient(response: httpx.Response) -> bool:
    """5xx, plus the 400 `EOF` a resident runner returns when `num_ctx` changes."""
    if response.status_code >= 500:
        return True
    return response.status_code == 400 and "EOF" in response.text


def _seconds(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else nanoseconds / 1e9


class OllamaClient:
    """One shared `httpx.AsyncClient` over Ollama's HTTP API.

    `transport` exists so tests drive every endpoint through
    `httpx.MockTransport` instead of a running Ollama.
    """

    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        *,
        timeout: httpx.Timeout,
        payload: dict | None = None,
    ) -> httpx.Response:
        response = await self._client.request(
            method, f"{self._base_url}{path}", json=payload, timeout=timeout
        )
        if response.status_code == 400 and _OVERFLOW_MARKER in response.text:
            raise ContextOverflow(f"{path}: {response.text.strip()}")
        response.raise_for_status()
        return response

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: httpx.Timeout,
        payload: dict | None = None,
    ) -> httpx.Response:
        """`_send` with retries on transient failures; ContextOverflow is not one."""
        for attempt in range(1, _MAX_ATTEMPTS):
            try:
                return await self._send(method, path, timeout=timeout, payload=payload)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            except httpx.HTTPStatusError as exc:
                if not _is_transient(exc.response):
                    raise
                reason = f"HTTP {exc.response.status_code}: {exc.response.text.strip()}"
            backoff = _BASE_BACKOFF * 2 ** (attempt - 1)
            logger.warning(
                "%s %s failed (%s); retrying in %.1fs (attempt %d/%d)",
                method,
                path,
                reason,
                backoff,
                attempt,
                _MAX_ATTEMPTS,
            )
            await asyncio.sleep(backoff)
        return await self._send(method, path, timeout=timeout, payload=payload)

    async def embed(
        self, model: str, text: str, *, num_ctx: int, truncate: bool = True
    ) -> EmbedResponse:
        """Embed one text. `truncate=False` raises ContextOverflow instead of silently clipping."""
        response = await self._request(
            "POST",
            "/api/embed",
            timeout=_EMBED_TIMEOUT,
            payload={
                "model": model,
                "input": text,
                "truncate": truncate,
                "options": {"num_ctx": num_ctx},
            },
        )
        body = response.json()
        return EmbedResponse(
            vector=body["embeddings"][0],
            prompt_eval_count=body.get("prompt_eval_count"),
        )

    async def chat_json(
        self, model: str, messages: list[dict], schema: dict, *, options: dict
    ) -> ChatResult:
        """One non-streaming chat turn whose answer is decoded against `schema`."""
        started = time.perf_counter()
        response = await self._request(
            "POST",
            "/api/chat",
            timeout=_CHAT_TIMEOUT,
            payload={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": options,
            },
        )
        wall_clock = time.perf_counter() - started
        body = response.json()
        content = body["message"]["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{model} returned invalid JSON ({exc}): {content[:200]!r}") from exc
        total = _seconds(body.get("total_duration"))
        return ChatResult(
            data=data,
            prompt_eval_count=body.get("prompt_eval_count"),
            eval_count=body.get("eval_count"),
            # The prefix cache shows up here, not in prompt_eval_count, which
            # keeps reporting the full prompt (review, WP9).
            prompt_eval_seconds=_seconds(body.get("prompt_eval_duration")),
            seconds=total if total is not None else wall_clock,
        )

    async def tags(self) -> dict[str, TagInfo]:
        """Locally available models, keyed by both `name` and `name` without `:latest`."""
        response = await self._request("GET", "/api/tags", timeout=_META_TIMEOUT)
        infos: dict[str, TagInfo] = {}
        for entry in response.json().get("models", []):
            name = entry["name"]
            details = entry.get("details") or {}
            info = TagInfo(
                digest=entry.get("digest", ""),
                quantization=details.get("quantization_level"),
                context_length=details.get("context_length"),
            )
            infos[name] = info
            bare, _, tag = name.rpartition(":")
            if tag == "latest" and bare:
                infos[bare] = info
        return infos

    async def version(self) -> str:
        response = await self._request("GET", "/api/version", timeout=_META_TIMEOUT)
        return response.json()["version"]

    async def warm_embedder(self, model: str, num_ctx: int) -> None:
        """Pay the model load once, up front — with a real embed call.

        `/api/generate {"model": …}` (the prod warm mechanism) returns 400 for an
        embedding model (R-m2), so it cannot warm one.
        """
        await self._request(
            "POST",
            "/api/embed",
            timeout=_WARM_TIMEOUT,
            payload={
                "model": model,
                "input": "warm",
                "truncate": True,
                "options": {"num_ctx": num_ctx},
            },
        )

    async def warm_chat(self, model: str) -> None:
        await self._request(
            "POST", "/api/generate", timeout=_WARM_TIMEOUT, payload={"model": model}
        )
