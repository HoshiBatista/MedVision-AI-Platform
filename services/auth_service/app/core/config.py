from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Session
    session_cookie_name: str = "medvision_session"
    session_ttl_seconds: int = 86400  # 24 hours

    # App
    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True


settings = Settings()
