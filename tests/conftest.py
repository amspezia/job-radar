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


# ---------------------------------------------------------------------------
# EVALS database (eval/evals_db, eval/embedding). Neither fixture is autouse and
# neither ever touches the DATABASE_URL database: they borrow its server and
# credentials to create a throwaway `job_radar_evals_test_*` database, which
# `drop_database` is the only thing allowed to remove. Imports are function-local
# so that importing this conftest stays as cheap as it was.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def evals_db_url():
    """A freshly created and migrated throwaway EVALS database, dropped at teardown."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4

    from eval.evals_db import admin

    # job_radar.config, imported above via job_radar.db.base, has load_dotenv()'d .env.
    server = os.environ["DATABASE_URL"]
    url = admin.with_database(server, f"job_radar_evals_test_{uuid4().hex[:8]}")
    admin.ensure_database(url)
    try:
        # Alembic's env.py calls asyncio.run(), which refuses to run inside the loop
        # pytest-asyncio may already have going — a worker thread keeps it clear of it.
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(admin.migrate, url).result()
        yield url
    finally:
        admin.drop_database(url)


@pytest_asyncio.fixture
async def evals_session(evals_db_url: str):
    """A session on an empty EVALS database — every table is truncated first."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from eval.embedding import models  # noqa: F401  (registers the embedding_* tables)
    from eval.evals_db.admin import database_name
    from eval.evals_db.base import EvalsBase, make_engine

    assert "_test_" in database_name(evals_db_url), "refusing to truncate a real database"
    engine = make_engine(evals_db_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            tables = ", ".join(f'"{t.name}"' for t in EvalsBase.metadata.sorted_tables)
            await session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
            await session.commit()
            yield session
    finally:
        await engine.dispose()
