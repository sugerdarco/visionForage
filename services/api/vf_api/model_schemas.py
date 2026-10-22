"""Public registry metadata, separate from private artifact storage records."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel


class ModelResponse(BaseModel):
    id: UUID
    project_id: UUID
    job_id: UUID
    version: int
    task_type: Literal["classification", "detection", "anomaly"]
    model_arch: str
    dataset_version_id: UUID
    dataset_version: int
    input_size: int
    is_promoted: bool
    created_at: datetime
    updated_at: datetime


class ModelDetail(ModelResponse):
    class_map: dict[str, int]
    hyperparams: dict[str, Any]


class ModelPage(BaseModel):
    items: list[ModelResponse]
    total: int
    page: int
    page_size: int
