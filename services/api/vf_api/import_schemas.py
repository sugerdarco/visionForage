"""Bounded ZIP import capabilities and public completion summaries."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


class InvalidDataset(ValueError):
    """A bounded, user-facing dataset validation failure."""


def archive_path(value: str) -> str:
    if (
        not value
        or len(value) > 500
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(char) < 32 for char in value)
        or "\\" in value
        or ":" in value
    ):
        raise InvalidDataset(
            "Archive paths must be relative POSIX paths without traversal"
        )
    return value


class ImportClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["imagefolder", "yolo", "coco"]
    size: int = Field(strict=True, gt=0, le=MAX_ARCHIVE_BYTES)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$", min_length=64, max_length=64)
    root: str = Field(default="", max_length=200)
    annotations_file: str = Field(default="annotations.json", max_length=200)

    @field_validator("root", "annotations_file")
    @classmethod
    def safe_paths(cls, value: str, info) -> str:
        # Keep even escaped Unicode paths comfortably within the grant token limit.
        if len(value.encode("utf-8")) > 200:
            raise ValueError("Import option paths cannot exceed 200 UTF-8 bytes")
        return (
            value if info.field_name == "root" and value == "" else archive_path(value)
        )


class ImportedImage(BaseModel):
    path: str
    image_id: UUID
    action: Literal["created", "labeled", "unchanged", "duplicate"]


class ImportResponse(BaseModel):
    project_id: UUID
    created_images: int
    labeled_images: int
    unchanged_images: int
    duplicate_files: int
    created_classes: int
    items: list[ImportedImage]
