from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"

    # Report LLM served by a local Ollama instance (on-prem, no external API).
    ollama_url: str = "http://ollama:11434"
    # Default: OpenBioLLM-8B (Q8_0 GGUF) pulled from Hugging Face. Override with any
    # Ollama model tag — e.g. a tiny one for CPU/CI (LLM_MODEL_NAME=qwen2.5:0.5b).
    llm_model_name: str = "hf.co/mradermacher/Llama3-OpenBioLLM-8B-GGUF:Q8_0"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.4
    llm_top_k: int = 50
    llm_top_p: float = 0.92
    llm_request_timeout: float = 300.0  # one generation call
    llm_pull_timeout: float = 3600.0    # first-run weight download

    otel_traces_enabled: bool = False
    otel_exporter_otlp_endpoint: str = ""

    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True
    # When False, the app does not auto-create tables on startup (Alembic owns the schema).
    auto_create_tables: bool = True


settings = Settings()
