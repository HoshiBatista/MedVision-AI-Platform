from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"
    sync_database_url: str = "postgresql+psycopg2://medvision:medvision@postgres:5432/medvision"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    storage_root: str = "/data/studies"

    # JWT (must match auth_service's secret/algorithm to validate Bearer tokens)
    jwt_secret_key: str = "change-me-in-production-use-a-long-random-secret"
    jwt_algorithm: str = "HS256"

    triton_http_url: str = "triton:8000"
    gradcam_service_url: str = "http://gradcam_service:8004"

    inference_conf_threshold: float = 0.25
    inference_iou_threshold: float = 0.45

    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True
    # When False, the app does not auto-create tables on startup (Alembic owns the schema).
    auto_create_tables: bool = True


settings = Settings()
