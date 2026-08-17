from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    ollama_base_url: str
    embedding_model: str
    generation_model: str
    # Optional smaller/faster model for ingest-time field extraction.
    # Defaults to generation_model when unset — extraction is a simple
    # classification task that a 3B model handles as well as a 7B one.
    extraction_model: str | None = None
    # Optional override for fit analysis, the dominant cost of a fit run
    # (~800 output tokens per job, decode-bound). Defaults to generation_model.
    # A smaller model here is faster, not better — measure fit agreement against
    # the current model before adopting one.
    fit_model: str | None = None
    # Selects the LLMProvider implementation (see adapters/providers.py).
    # "ollama" is the only implementation today.
    llm_provider: str = "ollama"
    # Read but not yet used by a real provider — Phase E wires these into a
    # second LLMProvider for the paid quality pass.
    llm_api_key: str | None = None
    llm_api_model: str | None = None


settings = Settings()
