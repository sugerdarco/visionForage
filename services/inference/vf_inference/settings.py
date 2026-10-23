"""A single verified bundle and a hashed endpoint key; no control-plane secrets."""

from pathlib import Path
import re
from uuid import UUID

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_", extra="ignore", hide_input_in_errors=True
    )

    bundle_dir: Path
    model_id: UUID
    model_version: int = Field(ge=1)
    slug: str = Field(pattern=r"^[a-z][a-z0-9-]{0,62}$")
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    api_key_sha256: SecretStr
    cpu_threads: int = Field(default=2, ge=1, le=8)

    @field_validator("api_key_sha256")
    @classmethod
    def validate_key_hash(cls, value):
        if not re.fullmatch(r"[a-f0-9]{64}", value.get_secret_value()):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return value
