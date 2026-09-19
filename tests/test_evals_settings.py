"""The one guard that stops the harness from pointing its own tooling at production.

`job_radar.config` calls `load_dotenv()` at import, so in a pytest process the real `.env`
values sit in `os.environ`; every case below passes explicit values and `_env_file=None` so
it tests the validator and not the developer's machine.
"""

import pytest
from pydantic import ValidationError

from eval.evals_db.settings import EvalsSettings

PROD = "postgresql+psycopg://job_radar:job_radar@localhost:5432/job_radar"
EVALS = "postgresql+psycopg://job_radar:job_radar@localhost:5432/job_radar_evals"


def _settings(evals: str, prod: str | None) -> EvalsSettings:
    return EvalsSettings(_env_file=None, evals_database_url=evals, database_url=prod)


def test_a_separate_database_is_accepted() -> None:
    settings = _settings(EVALS, PROD)

    assert settings.evals_database_url == EVALS
    assert settings.database_url == PROD
    assert settings.ollama_base_url == "http://localhost:11434"


def test_the_production_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="production database"):
        _settings(PROD, PROD)


def test_the_production_database_name_is_rejected_on_another_server() -> None:
    # Same database name, different host and credentials: still refused, because the two
    # URLs are far more likely to be the same database than a deliberate namesake.
    with pytest.raises(ValidationError, match="production database"):
        _settings("postgresql+psycopg://other:other@db.internal:5432/job_radar", PROD)


def test_production_may_be_unset() -> None:
    # Everything but `import-topic` and `verify` runs without a production URL at all.
    assert _settings(EVALS, None).database_url is None
