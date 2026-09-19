"""Embedder specs, the Ollama embedder's truncation policy, and the registry."""

import json
from collections.abc import Callable

import httpx
import numpy as np
import pytest

from eval.embedding.embedders.base import EmbedderSpec, apply_dim, fingerprint
from eval.embedding.embedders.ollama import OllamaEmbedder
from eval.embedding.embedders.registry import build_embedder, load_specs
from eval.llm.ollama import OllamaClient

_MODEL = "qwen3-embedding:0.6b"
_SPEC = EmbedderSpec(name="qwen3", model=_MODEL, mrl=True)
_DIGEST = "sha256:7f1f6b"


def _tags_body(name: str = _MODEL) -> dict:
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "digest": _DIGEST,
                "details": {"quantization_level": "Q8_0", "context_length": 32768},
            }
        ]
    }


def _ollama(
    embed: Callable[[dict], httpx.Response], tags: dict | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """A MockTransport handler routing /api/tags, /api/version and /api/embed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_tags_body() if tags is None else tags)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.30.11"})
        if request.url.path == "/api/embed":
            return embed(json.loads(request.content))
        raise AssertionError(f"unexpected request to {request.url.path}")

    return handler


def _embedder(handler: Callable[[httpx.Request], httpx.Response], spec=_SPEC) -> OllamaEmbedder:
    return OllamaEmbedder(
        spec, OllamaClient("http://ollama.test:11435", httpx.MockTransport(handler))
    )


# --- EmbedderSpec ------------------------------------------------------------


def test_dim_requires_mrl():
    with pytest.raises(ValueError, match="requires mrl"):
        EmbedderSpec(name="x", model="m", dim=768)
    assert EmbedderSpec(name="x", model="m", mrl=True, dim=768).dim == 768


def test_hyde_prefix_defaults_to_the_document_prefix():
    doc_only = EmbedderSpec(name="x", model="m", doc_prefix="search_document: ")
    assert doc_only.effective_hyde_prefix == "search_document: "

    query_side = EmbedderSpec(
        name="x", model="m", doc_prefix="search_document: ", hyde_prefix="search_query: "
    )
    assert query_side.effective_hyde_prefix == "search_query: "

    # An explicit empty prefix is a choice, not "unset".
    assert (
        EmbedderSpec(name="x", model="m", doc_prefix="d: ", hyde_prefix="").effective_hyde_prefix
        == ""
    )


# --- apply_dim ---------------------------------------------------------------


def test_apply_dim_none_passes_the_vector_through():
    vec = np.array([0.6, 0.8], dtype=np.float32)
    assert apply_dim(vec, None) is vec


def test_apply_dim_slices_then_renormalizes():
    vec = np.array([3.0, 4.0, 12.0], dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    out = apply_dim(vec, 2)

    assert out.shape == (2,)
    assert out.dtype == np.float32
    assert float(np.linalg.norm(out)) == pytest.approx(1.0)
    assert out == pytest.approx(np.array([0.6, 0.8], dtype=np.float32))


def test_apply_dim_rejects_widening_and_zero_slices():
    with pytest.raises(ValueError, match="cannot truncate"):
        apply_dim(np.ones(4, dtype=np.float32), 8)
    with pytest.raises(ValueError, match="all zero"):
        apply_dim(np.array([0.0, 0.0, 1.0], dtype=np.float32), 2)


# --- fingerprint -------------------------------------------------------------


def test_fingerprint_is_the_documented_digest():
    import hashlib

    expected = hashlib.sha256(b"ollama|sha256:abc|8192").hexdigest()
    assert fingerprint("ollama", "sha256:abc", 8192) == expected
    assert fingerprint("ollama", "sha256:abc", 2048) != expected
    assert fingerprint("ollama", "sha256:def", 8192) != expected


# --- OllamaEmbedder ----------------------------------------------------------


async def test_ready_points_at_the_pull_command_when_the_tag_is_missing():
    embedder = _embedder(_ollama(lambda _: httpx.Response(200), tags=_tags_body("bge-m3:latest")))

    with pytest.raises(RuntimeError, match=f"ollama pull {_MODEL}"):
        await embedder.ready()


async def test_describe_needs_ready_first():
    embedder = _embedder(_ollama(lambda _: httpx.Response(200)))

    with pytest.raises(RuntimeError, match="call ready"):
        embedder.describe()


async def test_ready_records_the_runtime_and_warms_the_model():
    payloads: list[dict] = []

    def embed(payload: dict) -> httpx.Response:
        payloads.append(payload)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]], "prompt_eval_count": 1})

    embedder = _embedder(_ollama(embed))
    await embedder.ready()

    assert payloads == [
        {"model": _MODEL, "input": "warm", "truncate": True, "options": {"num_ctx": 8192}}
    ]
    assert embedder.describe() == {
        "model": _MODEL,
        "digest": _DIGEST,
        "quantization": "Q8_0",
        "runtime_version": "0.30.11",
        "num_ctx": 8192,
        "fingerprint": fingerprint("ollama", _DIGEST, 8192),
    }


async def test_a_fitting_document_is_embedded_once_and_not_marked_truncated():
    payloads: list[dict] = []

    def embed(payload: dict) -> httpx.Response:
        payloads.append(payload)
        return httpx.Response(200, json={"embeddings": [[0.6, 0.8]], "prompt_eval_count": 311})

    embedder = _embedder(_ollama(embed))
    await embedder.ready()
    payloads.clear()

    result = await embedder.embed("a short posting")

    assert [p["truncate"] for p in payloads] == [False]
    assert result.truncated is False
    assert result.n_tokens == 311
    assert result.vector.dtype == np.float32
    assert result.vector == pytest.approx(np.array([0.6, 0.8], dtype=np.float32))


async def test_an_over_long_document_is_retried_truncated_and_flagged():
    payloads: list[dict] = []

    def embed(payload: dict) -> httpx.Response:
        payloads.append(payload)
        if payload["truncate"] is False:
            return httpx.Response(
                400, json={"error": "the input length exceeds the context length"}
            )
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]], "prompt_eval_count": 2048})

    embedder = _embedder(_ollama(embed))
    await embedder.ready()
    payloads.clear()

    result = await embedder.embed("x" * 100_000)

    assert [p["truncate"] for p in payloads] == [False, True]
    assert result.truncated is True
    assert result.n_tokens == 2048


async def test_vectors_are_cached_at_native_dimension():
    """`dim` is a ranking-time slice, so a d768 spec reuses the full vectors."""

    def embed(_: dict) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.5] * 1024]})

    spec = EmbedderSpec(name="qwen3-d768", model=_MODEL, mrl=True, dim=768)
    embedder = _embedder(_ollama(embed), spec=spec)
    await embedder.ready()

    result = await embedder.embed("a posting")

    assert result.vector.shape == (1024,)


# --- registry ----------------------------------------------------------------


def _toml(tmp_path, body: str):
    path = tmp_path / "embedders.toml"
    path.write_text(body)
    return path


_ONE = """
[[embedder]]
name = "a"
model = "m"
incumbent = true
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "no \\[\\[embedder\\]\\] tables"),
        (_ONE + '\n[[embedder]]\nname = "b"\nmodel = "m"\nincumbent = true\n', "exactly one"),
        ('[[embedder]]\nname = "a"\nmodel = "m"\n', "exactly one"),
        (_ONE + '\n[[embedder]]\nname = "a"\nmodel = "m"\n', "duplicate embedder name"),
        (_ONE + '\n[[embedder]]\nname = "b"\nmodel = "m"\nbackend = "hf"\n', "unknown backend"),
        (_ONE + '\n[[embedder]]\nname = "b"\nmodel = "m"\ndim = 768\n', "requires mrl"),
        (_ONE + '\n[[embedder]]\nname = "b"\nmodel = "m"\ntemperature = 0.5\n', "unknown key"),
    ],
)
def test_registry_rejects_bad_configurations(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_specs(_toml(tmp_path, body))


def test_registry_loads_a_valid_file(tmp_path):
    specs = load_specs(_toml(tmp_path, _ONE))
    assert [spec.name for spec in specs] == ["a"]
    assert specs[0].backend == "ollama"


def test_the_committed_file_holds_the_six_candidates():
    specs = load_specs()
    by_name = {spec.name: spec for spec in specs}

    assert list(by_name) == [
        "nomic-v1.5",
        "nomic-v1.5-hyde-query",
        "qwen3-0.6b-noinst",
        "qwen3-0.6b-instruct",
        "qwen3-0.6b-instruct-d768",
        "bge-m3",
    ]
    assert [spec.name for spec in specs if spec.incumbent] == ["nomic-v1.5"]
    assert all(spec.num_ctx == 8192 for spec in specs)
    assert by_name["nomic-v1.5"].doc_prefix == "search_document: "
    assert by_name["nomic-v1.5"].effective_hyde_prefix == "search_document: "
    assert by_name["nomic-v1.5-hyde-query"].effective_hyde_prefix == "search_query: "
    # Qwen3's official template: no space after "Query:".
    assert by_name["qwen3-0.6b-instruct"].effective_hyde_prefix == (
        "Instruct: Given a job description, retrieve similar job postings\nQuery:"
    )
    assert by_name["qwen3-0.6b-instruct"].doc_prefix == ""
    assert by_name["qwen3-0.6b-instruct-d768"].dim == 768
    assert by_name["bge-m3"].effective_hyde_prefix == ""


def test_build_embedder_returns_an_ollama_embedder():
    client = OllamaClient("http://ollama.test:11435")
    embedder = build_embedder(_SPEC, client)

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.spec is _SPEC
