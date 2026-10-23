"""Database configuration shared by migrations and the application."""

from urllib.parse import quote
from pathlib import Path

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
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
    s3_artifacts_bucket: str = "vf-artifacts"
    s3_predictions_bucket: str = "vf-predictions"

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


class DeploymentSettings(EnvironmentSettings):
    prediction_log_dir: Path = Path("artifacts/predictions")
    deployment_network: str = Field(
        default="visionforge_inference", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$"
    )
    deployment_gateway_url: AnyHttpUrl = "http://127.0.0.1:8080"
    deployment_cors_origins: list[AnyHttpUrl] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    deployment_image: str = "visionforge-inference:0.1.0"
    deployment_work_dir: Path = Path("artifacts/deployments")
    deployment_namespace: str = Field(
        default="visionforge", pattern=r"^[a-z][a-z0-9-]{0,30}$"
    )
    deployment_max_active: int = Field(default=2, ge=1, le=8)
    deployment_cpu_threads: int = Field(default=2, ge=1, le=8)
    deployment_memory_mb: int = Field(default=1536, ge=512, le=8192)
    deployment_ready_seconds: int = Field(default=90, ge=5, le=300)

    @field_validator("deployment_gateway_url")
    @classmethod
    def gateway_origin(cls, value):
        if (
            value.username
            or value.password
            or value.query
            or value.fragment
            or value.path not in (None, "/")
        ):
            raise ValueError(
                "Use an HTTP origin without credentials, path, query or fragment"
            )
        return value
