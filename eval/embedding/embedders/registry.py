"""Reads and validates `eval/embedders.toml`, and builds an embedder from a spec."""

import tomllib
from dataclasses import fields
from pathlib import Path

from eval.embedding.embedders.base import Embedder, EmbedderSpec
from eval.embedding.embedders.ollama import OllamaEmbedder
from eval.llm.ollama import OllamaClient

# Located relative to this file, so the harness works from any directory.
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "embedders.toml"
_BACKENDS = ("ollama",)
_KEYS = {field.name for field in fields(EmbedderSpec)}


def load_specs(path: Path | None = None) -> list[EmbedderSpec]:
    source = _DEFAULT_PATH if path is None else path
    with source.open("rb") as handle:
        entries = tomllib.load(handle).get("embedder", [])
    if not entries:
        raise ValueError(f"{source}: no [[embedder]] tables")

    specs = []
    for entry in entries:
        unknown = sorted(set(entry) - _KEYS)
        if unknown:
            raise ValueError(
                f"{source}: unknown key(s) {unknown} in embedder {entry.get('name')!r}"
            )
        spec = EmbedderSpec(**entry)
        if spec.backend not in _BACKENDS:
            raise ValueError(
                f"{spec.name}: unknown backend {spec.backend!r} (known: {', '.join(_BACKENDS)})"
            )
        specs.append(spec)

    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{source}: duplicate embedder name(s) {duplicates}")
    # The incumbent is the baseline every comparison and both parity checks are
    # stated against, so "which one" can never be ambiguous.
    incumbents = [spec.name for spec in specs if spec.incumbent]
    if len(incumbents) != 1:
        raise ValueError(
            f"{source}: exactly one embedder must set incumbent = true, found {incumbents}"
        )
    return specs


def build_embedder(spec: EmbedderSpec, client: OllamaClient) -> Embedder:
    """One backend today; a second one is a branch here."""
    return OllamaEmbedder(spec, client)
