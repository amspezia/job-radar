import pytest

from job_radar.adapters.embeddings import embed


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


async def test_embed_returns_first_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def make_client(*args: object, **kwargs: object) -> _FakeClient:
        client = _FakeClient()
        captured["client"] = client
        return client

    monkeypatch.setattr("job_radar.adapters.embeddings.httpx.AsyncClient", make_client)

    vector = await embed("senior backend engineer")

    assert vector == [0.1, 0.2, 0.3]  # unwraps embeddings[0], not the whole list
    assert captured["client"].posted["json"]["input"] == "senior backend engineer"
