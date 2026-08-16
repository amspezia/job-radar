import httpx
import pytest

from job_radar.adapters.retry import with_retry


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://example.com")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("job_radar.adapters.retry.asyncio.sleep", _instant_sleep)


async def test_succeeds_on_first_attempt_without_retrying() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await with_retry(_call, label="t") == "ok"
    assert calls == 1


async def test_retries_transient_5xx_then_succeeds() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _status_error(503)
        return "ok"

    assert await with_retry(_call, label="t") == "ok"
    assert calls == 3


async def test_retries_timeout_and_connect_errors() -> None:
    calls = 0
    errors = [httpx.TimeoutException("slow"), httpx.ConnectError("refused")]

    async def _call() -> str:
        nonlocal calls
        calls += 1
        if calls <= len(errors):
            raise errors[calls - 1]
        return "ok"

    assert await with_retry(_call, label="t") == "ok"
    assert calls == 3


async def test_does_not_retry_4xx() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        raise _status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await with_retry(_call, label="t")
    assert calls == 1


async def test_does_not_retry_non_httpx_exceptions() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError, match="not transient"):
        await with_retry(_call, label="t")
    assert calls == 1


async def test_gives_up_after_max_attempts() -> None:
    calls = 0

    async def _call() -> str:
        nonlocal calls
        calls += 1
        raise _status_error(500)

    with pytest.raises(httpx.HTTPStatusError):
        await with_retry(_call, label="t")
    assert calls == 3
