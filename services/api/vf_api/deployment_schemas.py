"""Public deployment state; raw endpoint keys are returned only on creation."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DeploymentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    slug: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,62}$")
    confidence_threshold: float = Field(default=0.6, ge=0, le=1)


class DeploymentResponse(BaseModel):
    id: UUID
    model_id: UUID
    project_id: UUID
    model_version: int
    slug: str
    status: Literal["starting", "running", "stopped", "failed"]
    confidence_threshold: float
    endpoint_url: str | None = None
    runtime_checked: bool = False
    created_at: datetime
    updated_at: datetime


class DeploymentCreated(DeploymentResponse):
    api_key: str | None


class DeploymentPage(BaseModel):
    items: list[DeploymentResponse]
    total: int
    page: int
    page_size: int
