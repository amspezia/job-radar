import asyncio

import pytest

from job_radar.adapters.sources.base import NormalizedJob, SourceAdapter
from job_radar.ingest import runner


class _NamedAdapter(SourceAdapter):
    source_type = "test"

    def __init__(self, source: str, *, on_fetch: asyncio.Event | None = None) -> None:
        self.source = source
        self._on_fetch = on_fetch

    async def fetch(self) -> list[dict]:
        if self._on_fetch is not None:
            self._on_fetch.set()
        return []

    def map(self, raw: dict) -> NormalizedJob:  # pragma: no cover - never called
        raise NotImplementedError


async def test_run_all_ingestion_isolates_a_failing_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    async def fake_run_ingestion(
        adapter: SourceAdapter,
        session: object,
        ingested_via: str,
        raw_postings: list[dict] | None = None,
    ) -> None:
        attempted.append(adapter.source)
        if adapter.source == "boom":
            raise RuntimeError("source down")

    monkeypatch.setattr(runner, "run_ingestion", fake_run_ingestion)
    monkeypatch.setattr(runner, "ENABLED_ADAPTERS", [_NamedAdapter("boom"), _NamedAdapter("ok")])

    # Must not raise: a failing adapter is logged and skipped, the rest still run.
    await runner.run_all_ingestion(session=object(), ingested_via="scheduler")

    assert attempted == ["boom", "ok"]


async def test_run_all_ingestion_isolates_a_failing_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    class _BrokenFetchAdapter(_NamedAdapter):
        async def fetch(self) -> list[dict]:
            raise RuntimeError("board down")

    async def fake_run_ingestion(
        adapter: SourceAdapter,
        session: object,
        ingested_via: str,
        raw_postings: list[dict] | None = None,
    ) -> None:
        attempted.append((adapter.source, raw_postings))

    monkeypatch.setattr(runner, "run_ingestion", fake_run_ingestion)
    monkeypatch.setattr(
        runner, "ENABLED_ADAPTERS", [_BrokenFetchAdapter("broken"), _NamedAdapter("ok")]
    )

    # A fetch failure must not abort the run — the adapter still gets
    # processed (with no postings) and the next adapter still runs.
    await runner.run_all_ingestion(session=object(), ingested_via="scheduler")

    assert attempted == [("broken", []), ("ok", [])]


async def test_run_all_ingestion_overlaps_next_fetch_with_current_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b_fetch_started = asyncio.Event()

    async def fake_run_ingestion(
        adapter: SourceAdapter,
        session: object,
        ingested_via: str,
        raw_postings: list[dict] | None = None,
    ) -> None:
        if adapter.source == "a":
            # Only resolves once b's fetch has actually started, proving the
            # two run concurrently rather than one after the other.
            await asyncio.wait_for(b_fetch_started.wait(), timeout=1)

    monkeypatch.setattr(runner, "run_ingestion", fake_run_ingestion)
    monkeypatch.setattr(
        runner,
        "ENABLED_ADAPTERS",
        [_NamedAdapter("a"), _NamedAdapter("b", on_fetch=b_fetch_started)],
    )

    await runner.run_all_ingestion(session=object(), ingested_via="scheduler")
