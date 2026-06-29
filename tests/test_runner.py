import pytest

from job_radar.adapters.sources.base import NormalizedJob, SourceAdapter
from job_radar.ingest import runner


class _NamedAdapter(SourceAdapter):
    source_type = "test"

    def __init__(self, source: str) -> None:
        self.source = source

    async def fetch(self) -> list[dict]:
        return []

    def map(self, raw: dict) -> NormalizedJob:  # pragma: no cover - never called
        raise NotImplementedError


async def test_run_all_ingestion_isolates_a_failing_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    async def fake_run_ingestion(
        adapter: SourceAdapter, session: object, ingested_via: str
    ) -> None:
        attempted.append(adapter.source)
        if adapter.source == "boom":
            raise RuntimeError("source down")

    monkeypatch.setattr(runner, "run_ingestion", fake_run_ingestion)
    monkeypatch.setattr(runner, "ENABLED_ADAPTERS", [_NamedAdapter("boom"), _NamedAdapter("ok")])

    # Must not raise: a failing adapter is logged and skipped, the rest still run.
    await runner.run_all_ingestion(session=object(), ingested_via="scheduler")

    assert attempted == ["boom", "ok"]
