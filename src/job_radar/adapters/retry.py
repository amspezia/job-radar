import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BASE_BACKOFF = 1.0  # seconds; doubles each retry: 1s, 2s


def _is_transient(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TimeoutException | httpx.ConnectError)


async def with_retry[T](
    call: Callable[[], Awaitable[T]],
    *,
    label: str,
    on_attempt: Callable[[int], None] | None = None,
) -> T:
    """Retry `call` up to _MAX_ATTEMPTS times on transient httpx failures only.

    Transient: timeouts, connection errors, and 5xx responses — the shape of a
    local-Ollama hiccup under concurrent load. Never retries 4xx (a bad request
    retrying won't fix) or any non-httpx exception (e.g. schema validation,
    TruncatedGeneration — a context-window problem, not a transient one).

    `on_attempt`, if given, is called once with the 1-based attempt number that
    actually succeeded — e.g. for a Langfuse trace to record retry count without
    this function needing to know anything about tracing itself.
    """
    last_exc: httpx.HTTPError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            result = await call()
            if on_attempt is not None:
                on_attempt(attempt)
            return result
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            if not _is_transient(exc) or attempt == _MAX_ATTEMPTS:
                raise
            last_exc = exc
            backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
            logger.warning(
                "%s: attempt %d/%d failed (%s); retrying in %.0fs",
                label,
                attempt,
                _MAX_ATTEMPTS,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)
    raise last_exc  # type: ignore[misc]  # unreachable: loop always returns or raises
