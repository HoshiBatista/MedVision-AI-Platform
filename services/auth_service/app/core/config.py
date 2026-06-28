from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://medvision:medvision@postgres:5432/medvision"

    # JWT
    jwt_secret_key: str = "change-me-in-production-use-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    # Long-lived, DB-backed refresh tokens (opaque; rotated on use, revocable).
    refresh_token_expire_days: int = 7
    # Single-use password-reset tokens (opaque; delivered out-of-band).
    password_reset_expire_minutes: int = 30

    # Admin seed
    admin_username: str = "admin"
    admin_password: str = "admin"

    # App
    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True
    # When False, the app does not auto-create tables on startup (Alembic owns the schema).
    auto_create_tables: bool = True


settings = Settings()
