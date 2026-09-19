"""The harness's Ollama client, driven entirely through httpx.MockTransport.

Every response shape here is one Ollama 0.30.11 actually returned during the
design review; no test starts a server.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from eval.llm import ollama as ollama_module
from eval.llm.ollama import ContextOverflow, OllamaClient

_BASE = "http://ollama.test:11435"


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    """Retries are exercised for real; their sleeps are not."""
    monkeypatch.setattr(ollama_module, "_BASE_BACKOFF", 0.0)


@pytest.fixture
async def make_client():
    created: list[OllamaClient] = []

    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
        client = OllamaClient(_BASE, transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    yield _make
    for client in created:
        await client.aclose()


def _recorder(
    responses: list[httpx.Response] | Callable[[httpx.Request], httpx.Response],
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    """Return (captured requests, handler). `responses` is a list or a function."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if callable(responses):
            return responses(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return seen, handler


async def test_embed_request_shape_and_response(make_client):
    seen, handler = _recorder(
        [httpx.Response(200, json={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 7})]
    )
    client = make_client(handler)

    response = await client.embed("bge-m3", "hello", num_ctx=8192, truncate=False)
    await client.embed("bge-m3", "hello", num_ctx=2048)

    assert str(seen[0].url) == f"{_BASE}/api/embed"
    assert json.loads(seen[0].content) == {
        "model": "bge-m3",
        "input": "hello",
        "truncate": False,
        "options": {"num_ctx": 8192},
    }
    assert json.loads(seen[1].content)["truncate"] is True
    assert json.loads(seen[1].content)["options"] == {"num_ctx": 2048}
    assert response.vector == [0.1, 0.2]
    assert response.prompt_eval_count == 7


async def test_over_long_input_raises_context_overflow_without_retrying(make_client):
    seen, handler = _recorder(
        [httpx.Response(400, json={"error": "the input length exceeds the context length"})]
    )
    client = make_client(handler)

    with pytest.raises(ContextOverflow, match="exceeds the context length"):
        await client.embed("nomic-embed-text", "x" * 100_000, num_ctx=8192, truncate=False)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500, text="internal error"),
        httpx.Response(503, text="model is loading"),
        # Observed when a resident runner is asked for a different num_ctx.
        httpx.Response(
            400, json={"error": "an error was encountered while running the model: EOF"}
        ),
    ],
)
async def test_transient_failures_are_retried(make_client, failure):
    ok = httpx.Response(200, json={"embeddings": [[1.0]], "prompt_eval_count": 1})
    seen, handler = _recorder([failure, ok])
    client = make_client(handler)

    response = await client.embed("bge-m3", "hi", num_ctx=8192)

    assert len(seen) == 2
    assert response.vector == [1.0]


async def test_timeouts_are_retried(make_client):
    ok = httpx.Response(200, json={"embeddings": [[1.0]]})
    calls = 0

    def responses(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("connect timed out")
        if calls == 2:
            raise httpx.ReadTimeout("read timed out")
        return ok

    seen, handler = _recorder(responses)
    client = make_client(handler)

    response = await client.embed("bge-m3", "hi", num_ctx=8192)

    assert len(seen) == 3
    assert response.prompt_eval_count is None


async def test_retries_are_capped(make_client):
    seen, handler = _recorder([httpx.Response(500, text="boom")])
    client = make_client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.embed("bge-m3", "hi", num_ctx=8192)
    assert len(seen) == 3


async def test_other_client_errors_are_not_retried(make_client):
    seen, handler = _recorder([httpx.Response(404, json={"error": 'model "nope" not found'})])
    client = make_client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.embed("nope", "hi", num_ctx=8192)
    assert len(seen) == 1


async def test_chat_json_request_shape_and_timings(make_client):
    seen, handler = _recorder(
        [
            httpx.Response(
                200,
                json={
                    "message": {"content": '{"reason": "close match", "grade": 2}'},
                    "prompt_eval_count": 2600,
                    "eval_count": 48,
                    "prompt_eval_duration": 3_350_000_000,
                    "total_duration": 4_000_000_000,
                },
            )
        ]
    )
    client = make_client(handler)
    schema = {"type": "object", "properties": {"grade": {"type": "integer"}}}

    result = await client.chat_json(
        "Gemma3:12b",
        [{"role": "user", "content": "grade this"}],
        schema,
        options={"temperature": 0, "num_ctx": 8192},
    )

    assert str(seen[0].url) == f"{_BASE}/api/chat"
    assert json.loads(seen[0].content) == {
        "model": "Gemma3:12b",
        "messages": [{"role": "user", "content": "grade this"}],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    assert result.data == {"reason": "close match", "grade": 2}
    assert result.prompt_eval_count == 2600
    assert result.eval_count == 48
    assert result.prompt_eval_seconds == pytest.approx(3.35)
    assert result.seconds == pytest.approx(4.0)


async def test_chat_json_without_durations_falls_back_to_wall_clock(make_client):
    _, handler = _recorder([httpx.Response(200, json={"message": {"content": "{}"}})])
    client = make_client(handler)

    result = await client.chat_json("Gemma3:12b", [], {}, options={})

    assert result.data == {}
    assert result.prompt_eval_seconds is None
    assert result.seconds >= 0.0


async def test_chat_json_rejects_invalid_json(make_client):
    _, handler = _recorder([httpx.Response(200, json={"message": {"content": "not json at all"}})])
    client = make_client(handler)

    with pytest.raises(ValueError, match="invalid JSON"):
        await client.chat_json("Gemma3:12b", [], {}, options={})


async def test_tag_names_resolve_with_and_without_latest(make_client):
    seen, handler = _recorder(
        [
            httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "bge-m3:latest",
                            "model": "bge-m3:latest",
                            "digest": "sha256:790764",
                            "details": {"quantization_level": "F16", "context_length": 8192},
                        },
                        {
                            "name": "Gemma3:12b",
                            "model": "Gemma3:12b",
                            "digest": "sha256:f4031a",
                            "details": {"quantization_level": "Q4_K_M"},
                        },
                    ]
                },
            )
        ]
    )
    client = make_client(handler)

    infos = await client.tags()

    assert str(seen[0].url) == f"{_BASE}/api/tags"
    assert set(infos) == {"bge-m3", "bge-m3:latest", "Gemma3:12b"}
    assert infos["bge-m3"] is infos["bge-m3:latest"]
    assert infos["bge-m3"].digest == "sha256:790764"
    assert infos["bge-m3"].quantization == "F16"
    assert infos["bge-m3"].context_length == 8192
    assert infos["Gemma3:12b"].context_length is None


async def test_version(make_client):
    seen, handler = _recorder([httpx.Response(200, json={"version": "0.30.11"})])
    client = make_client(handler)

    assert await client.version() == "0.30.11"
    assert str(seen[0].url) == f"{_BASE}/api/version"


async def test_warm_embedder_uses_embed_not_generate(make_client):
    seen, handler = _recorder([httpx.Response(200, json={"embeddings": [[0.0]]})])
    client = make_client(handler)

    await client.warm_embedder("bge-m3", 8192)

    # /api/generate returns 400 for an embedding model (R-m2).
    assert str(seen[0].url) == f"{_BASE}/api/embed"
    assert json.loads(seen[0].content) == {
        "model": "bge-m3",
        "input": "warm",
        "truncate": True,
        "options": {"num_ctx": 8192},
    }


async def test_warm_chat_uses_generate(make_client):
    seen, handler = _recorder([httpx.Response(200, json={"done": True})])
    client = make_client(handler)

    await client.warm_chat("Gemma3:12b")

    assert str(seen[0].url) == f"{_BASE}/api/generate"
    assert json.loads(seen[0].content) == {"model": "Gemma3:12b"}
