from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"
    redis_url: str = "redis://redis:6379/0"

    # Local filesystem storage root (mounted volume in Docker)
    storage_root: str = "/data/studies"

    # JWT (must match auth_service's secret/algorithm to validate Bearer tokens)
    jwt_secret_key: str = "change-me-in-production-use-a-long-random-secret"
    jwt_algorithm: str = "HS256"

    max_upload_size_mb: int = 512

    otel_traces_enabled: bool = False
    otel_exporter_otlp_endpoint: str = ""

    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True
    # When False, the app does not auto-create tables on startup (Alembic owns the schema).
    auto_create_tables: bool = True


settings = Settings()
