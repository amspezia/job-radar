import httpx
import pytest
from pydantic import BaseModel

from job_radar.adapters.generation import TruncatedGeneration, generate
from tests.conftest import FakeLangfuseClient


class _Schema(BaseModel):
    value: str


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _FakeClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.posted: dict | None = None

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self.posted = {"url": url, "json": json}
        return _FakeResponse({"done_reason": "stop", "message": {"content": '{"value": "ok"}'}})


def _make_client(captured: dict, client_cls: type = _FakeClient):
    def factory(*args: object, **kwargs: object) -> _FakeClient:
        client = client_cls()
        captured["client"] = client
        return client

    return factory


# fake_langfuse fixture is defined in conftest.py (autouse=True there — every
# test in the suite gets it automatically now, not just this file); imported
# here only for the type hint on the tests below that inspect its calls.


async def test_generate_parses_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    result = await generate("a prompt", _Schema)

    assert result == _Schema(value="ok")


async def test_generate_sends_schema_constrained_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    await generate("a prompt", _Schema)

    payload = captured["client"].posted["json"]
    assert payload["format"] == _Schema.model_json_schema()
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_ctx"] == 8192


async def test_generate_uses_explicit_model_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    await generate("a prompt", _Schema, model="custom-model")

    assert captured["client"].posted["json"]["model"] == "custom-model"


async def test_generate_raises_truncated_generation_on_length_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TruncatingClient(_FakeClient):
        async def post(self, url: str, json: dict) -> _FakeResponse:
            return _FakeResponse(
                {
                    "done_reason": "length",
                    "prompt_eval_count": 100,
                    "eval_count": 50,
                    "message": {"content": "{}"},
                }
            )

    captured: dict = {}
    monkeypatch.setattr(
        "job_radar.adapters.providers.httpx.AsyncClient",
        _make_client(captured, _TruncatingClient),
    )

    with pytest.raises(TruncatedGeneration):
        await generate("a prompt", _Schema)


async def test_generate_retries_transient_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class _FlakyClient(_FakeClient):
        async def post(self, url: str, json: dict) -> _FakeResponse:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                request = httpx.Request("POST", url)
                raise httpx.ConnectError("ollama unreachable", request=request)
            return await super().post(url, json)

    monkeypatch.setattr(
        "job_radar.adapters.providers.httpx.AsyncClient",
        _make_client({}, _FlakyClient),
    )

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("job_radar.adapters.retry.asyncio.sleep", _instant_sleep)

    result = await generate("a prompt", _Schema)

    assert result == _Schema(value="ok")
    assert attempts == 2


async def test_generate_creates_a_langfuse_generation_observation(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client({}))

    await generate("a prompt", _Schema, model="custom-model")

    assert len(fake_langfuse.observations) == 1
    obs = fake_langfuse.observations[0]
    assert obs["as_type"] == "generation"
    assert obs["model"] == "custom-model"
    assert obs["name"] == "generate._Schema"


async def test_generate_redacts_input_by_default(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client({}))
    prompt = "a prompt with secrets in it"

    await generate(prompt, _Schema)

    obs = fake_langfuse.observations[0]
    assert obs["input"] == {"chars": len(prompt)}
    assert "secrets" not in str(obs["input"])


async def test_generate_redacts_output_by_default(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client({}))

    await generate("a prompt", _Schema)

    obs = fake_langfuse.observations[0]
    assert obs["output_update"] == {"output": {"schema": "_Schema"}}


async def test_generate_enriches_current_generation_with_usage_and_retry_count(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    class _UsageClient(_FakeClient):
        async def post(self, url: str, json: dict) -> _FakeResponse:
            return _FakeResponse(
                {
                    "done_reason": "stop",
                    "prompt_eval_count": 42,
                    "eval_count": 7,
                    "message": {"content": '{"value": "ok"}'},
                }
            )

    monkeypatch.setattr(
        "job_radar.adapters.providers.httpx.AsyncClient", _make_client({}, _UsageClient)
    )

    await generate("a prompt", _Schema)

    assert fake_langfuse.generation_updates == [
        {"usage_details": {"input": 42, "output": 7}, "metadata": {"retry_attempts": 1}}
    ]


async def test_generate_records_retry_attempts_when_a_retry_occurred(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    attempts = 0

    class _FlakyClient(_FakeClient):
        async def post(self, url: str, json: dict) -> _FakeResponse:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise httpx.ConnectError("ollama unreachable", request=httpx.Request("POST", url))
            return await super().post(url, json)

    monkeypatch.setattr(
        "job_radar.adapters.providers.httpx.AsyncClient", _make_client({}, _FlakyClient)
    )

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("job_radar.adapters.retry.asyncio.sleep", _instant_sleep)

    await generate("a prompt", _Schema)

    assert fake_langfuse.generation_updates[0]["metadata"] == {"retry_attempts": 2}
