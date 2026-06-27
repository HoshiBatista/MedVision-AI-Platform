from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"

    # BioGPT model — swap to "microsoft/BioGPT-Large" for better quality
    llm_model_name: str = "microsoft/biogpt"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.4
    llm_top_k: int = 50
    llm_top_p: float = 0.92
    # "auto" → picks MPS on Apple Silicon, then CUDA, then CPU
    llm_device: str = "auto"

    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True
    # When False, the app does not auto-create tables on startup (Alembic owns the schema).
    auto_create_tables: bool = True


settings = Settings()
