"""Upload capabilities contain server-chosen keys and validated file claims."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from vf_api.storage import MAX_UPLOAD_BYTES


class FileClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: Literal["image/jpeg", "image/png"]
    size: int = Field(strict=True, gt=0, le=MAX_UPLOAD_BYTES)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$", min_length=64, max_length=64)


class PresignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[FileClaim] = Field(min_length=1, max_length=20)


class UploadPermission(BaseModel):
    upload_token: str
    upload_url: str
    headers: dict[str, str]
    expires_in: int


class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_token: SecretStr = Field(min_length=1, max_length=4096)


class ImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    width: int
    height: int
    sha256: str
    status: Literal["unlabeled", "labeled", "in_review"]
    created_at: datetime
    image_url: str
    thumbnail_url: str | None
    urls_expire_in: int


class ImagePage(BaseModel):
    items: list[ImageResponse]
    page: int
    page_size: int
    total: int
