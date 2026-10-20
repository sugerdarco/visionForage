"""Public authentication payloads; password hashes never enter response schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    email: EmailStr = Field(max_length=320)
    password: SecretStr = Field(min_length=12, max_length=72)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        # Account identities are case-insensitive, including the local part.
        return value.lower()

    @field_validator("password")
    @classmethod
    def check_password_bytes(cls, value: SecretStr) -> SecretStr:
        try:
            length = len(value.get_secret_value().encode("utf-8"))
        except UnicodeError:
            raise ValueError("Password must be valid UTF-8") from None
        if length > 72:
            raise ValueError("Password must fit in 72 UTF-8 bytes")
        return value


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    refresh_token: SecretStr = Field(min_length=1, max_length=4096)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    created_at: datetime


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class LoginResponse(AccessTokenResponse):
    refresh_token: str
    refresh_expires_in: int
