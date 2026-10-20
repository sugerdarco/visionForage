"""Database configuration shared by migrations and the application."""

from urllib.parse import quote

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )


class Settings(EnvironmentSettings):
    postgres_host: str = "127.0.0.1"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "visionforge"
    postgres_user: str = "visionforge"
    postgres_password: SecretStr = Field(min_length=1)

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


class APISettings(Settings):
    # Kept separate so database migrations do not require an authentication key.
    jwt_secret: SecretStr = Field(min_length=64)
    access_token_minutes: int = Field(default=15, ge=1, le=60)
    refresh_token_days: int = Field(default=7, ge=1, le=30)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )


class StorageSettings(EnvironmentSettings):
    minio_root_user: str = "visionforge"
    minio_root_password: SecretStr = Field(min_length=1)
    minio_api_port: int = Field(default=9000, ge=1, le=65535)
    s3_endpoint_url: AnyHttpUrl | None = None
    s3_public_endpoint_url: AnyHttpUrl | None = None
    s3_region: str = "us-east-1"
    s3_images_bucket: str = "vf-images"
    s3_datasets_bucket: str = "vf-datasets"

    @property
    def endpoint(self) -> str:
        return str(self.s3_endpoint_url or f"http://127.0.0.1:{self.minio_api_port}")

    @property
    def public_endpoint(self) -> str:
        return str(self.s3_public_endpoint_url or self.endpoint)


class QueueSettings(EnvironmentSettings):
    redis_host: str = "127.0.0.1"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_password: SecretStr = Field(min_length=1)

    @property
    def broker_url(self) -> str:
        password = quote(self.redis_password.get_secret_value(), safe="")
        return f"redis://:{password}@{self.redis_host}:{self.redis_port}/0"
