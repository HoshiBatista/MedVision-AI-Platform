from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"
    redis_url: str = "redis://redis:6379/0"

    # Local filesystem storage root (mounted volume in Docker)
    storage_root: str = "/data/studies"

    # Session (shared with auth_service — same Redis keys)
    session_cookie_name: str = "medvision_session"

    max_upload_size_mb: int = 512

    environment: str = "development"
    docs_enabled: bool = True


settings = Settings()
