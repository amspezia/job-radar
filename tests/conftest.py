import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from job_radar.db.base import async_session_factory


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    async with async_session_factory() as session:
        yield session


class FakeLangfuseObservation:
    def __init__(self, record: dict) -> None:
        self._record = record

    def update(self, **kwargs: object) -> None:
        self._record["output_update"] = kwargs

    def __enter__(self) -> "FakeLangfuseObservation":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class FakeLangfuseClient:
    """Fake for langfuse.get_client() — autouse (see fake_langfuse below) so no
    test can ever reach real Langfuse Cloud, regardless of what credentials
    happen to be present in the environment. Concretely fixes an incident: a
    test run without `env -i` sent synthetic test traces (e.g. the `_Schema`
    fake schema class from test_generation.py) into a real Langfuse Cloud
    project, indistinguishable from production traces in the UI.
    """

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.generation_updates: list[dict] = []

    def start_as_current_observation(self, **kwargs: object) -> FakeLangfuseObservation:
        record = dict(kwargs)
        self.observations.append(record)
        return FakeLangfuseObservation(record)

    def update_current_generation(self, **kwargs: object) -> None:
        self.generation_updates.append(kwargs)


@pytest.fixture(autouse=True)
def fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> FakeLangfuseClient:
    client = FakeLangfuseClient()
    for module in (
        "job_radar.adapters.generation",
        "job_radar.adapters.embeddings",
        "job_radar.adapters.providers",
        "job_radar.fit.analyze",
        "job_radar.retrieval.search",
        "eval.qrels",
    ):
        monkeypatch.setattr(f"{module}.get_client", lambda _client=client: _client)
    return client
