import httpx
import pytest

from job_radar.adapters.embeddings import embed
from tests.conftest import FakeLangfuseClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        # Ollama returns a list of vectors even for a single input.
        return {"embeddings": [[0.1, 0.2, 0.3]]}


class _FakeClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.posted: dict | None = None

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self.posted = {"url": url, "json": json}
        return _FakeResponse()


def _make_client(captured: dict, client_cls: type = _FakeClient):
    def factory(*args: object, **kwargs: object) -> _FakeClient:
        client = client_cls()
        captured["client"] = client
        return client

    return factory


# fake_langfuse fixture is defined in conftest.py (autouse=True there — every
# test in the suite gets it automatically now, not just this file); imported
# here only for the type hint on the tests below that inspect its calls.


async def test_embed_returns_first_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    vector = await embed("senior backend engineer", task="query")

    assert vector == [0.1, 0.2, 0.3]  # unwraps embeddings[0], not the whole list


async def test_embed_prepends_query_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    await embed("senior backend engineer", task="query")

    assert captured["client"].posted["json"]["input"] == "search_query: senior backend engineer"


async def test_embed_prepends_document_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    await embed("Backend Engineer\nBuild scalable APIs.", task="document")

    assert (
        captured["client"].posted["json"]["input"]
        == "search_document: Backend Engineer\nBuild scalable APIs."
    )


async def test_embed_sets_num_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client(captured))

    await embed("text", task="document")

    assert captured["client"].posted["json"]["options"]["num_ctx"] == 8192


async def test_embed_retries_transient_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class _FlakyClient(_FakeClient):
        async def post(self, url: str, json: dict) -> _FakeResponse:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise httpx.ConnectError("ollama unreachable", request=httpx.Request("POST", url))
            return await super().post(url, json)

    def factory(*args: object, **kwargs: object) -> _FlakyClient:
        return _FlakyClient()

    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", factory)

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("job_radar.adapters.retry.asyncio.sleep", _instant_sleep)

    vector = await embed("text", task="document")

    assert vector == [0.1, 0.2, 0.3]
    assert attempts == 2


async def test_embed_creates_a_langfuse_embedding_observation(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client({}))

    await embed("senior backend engineer", task="query")

    assert len(fake_langfuse.observations) == 1
    obs = fake_langfuse.observations[0]
    assert obs["as_type"] == "embedding"
    assert obs["name"] == "embed"
    assert obs["metadata"] == {"task": "query"}


async def test_embed_redacts_input_by_default(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client({}))

    await embed("some CV text", task="document")

    obs = fake_langfuse.observations[0]
    assert "chars" in obs["input"]
    assert "CV text" not in str(obs["input"])


async def test_embed_output_reports_dims_not_the_vector(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    monkeypatch.setattr("job_radar.adapters.providers.httpx.AsyncClient", _make_client({}))

    await embed("text", task="document")

    obs = fake_langfuse.observations[0]
    assert obs["output_update"] == {"output": {"dims": 3}}


async def test_embed_enriches_current_generation_with_usage_and_retry_count(
    monkeypatch: pytest.MonkeyPatch, fake_langfuse: FakeLangfuseClient
) -> None:
    class _UsageResponse(_FakeResponse):
        def json(self) -> dict:
            return {**super().json(), "prompt_eval_count": 4}

    class _UsageClient(_FakeClient):
        async def post(self, url: str, json: dict) -> _FakeResponse:
            self.posted = {"url": url, "json": json}
            return _UsageResponse()

    monkeypatch.setattr(
        "job_radar.adapters.providers.httpx.AsyncClient", _make_client({}, _UsageClient)
    )

    await embed("text", task="document")

    assert fake_langfuse.generation_updates == [
        {"usage_details": {"input": 4}, "metadata": {"retry_attempts": 1}}
    ]
