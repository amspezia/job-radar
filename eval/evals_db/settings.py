"""Configuration for the eval harness.

Deliberately independent of `job_radar.config`: importing that module instantiates the
production settings and, transitively, a read-write engine (review F13/R-M5). The harness
reads production through `prod_reader.py` only, so it needs the prod URL as a plain string
and nothing else from production code.

`job_radar.config` calls `load_dotenv()` at import, so inside a pytest process the `.env`
values are already in `os.environ`; tests construct `EvalsSettings` with explicit keyword
arguments and `_env_file=None` to keep that out of the way.
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class EvalsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # The database the harness owns: created, migrated, written and dropped freely.
    evals_database_url: str
    # Production. Optional because only `import-topic` and `verify` read it.
    database_url: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    @model_validator(mode="after")
    def _evals_database_is_not_production(self) -> "EvalsSettings":
        if self.database_url is None:
            return self
        if self.evals_database_url == self.database_url or (
            make_url(self.evals_database_url).database == make_url(self.database_url).database
        ):
            raise ValueError(
                "EVALS_DATABASE_URL points at the production database. The harness creates, "
                "migrates and truncates this database — it must be a separate one."
            )
        return self


@lru_cache
def get_settings() -> EvalsSettings:
    return EvalsSettings()
