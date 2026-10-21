"""Bounded training requests without importing the ML runtime."""

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    horizontal_flip: bool = False
    vertical_flip: bool = False


class ClassificationParameters(Parameters):
    epochs: int = Field(default=20, ge=1, le=1000)
    batch_size: int = Field(default=64, ge=1, le=256)
    patience: int = Field(default=5, ge=1, le=1000)
    learning_rate: float = Field(default=3e-4, ge=1e-8, le=1)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    label_smoothing: float = Field(default=0.1, ge=0, lt=1)


class DetectionParameters(Parameters):
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    patience: int = Field(default=20, ge=1, le=1000)
    image_size: int = Field(default=640, ge=64, le=1280)

    @field_validator("image_size")
    @classmethod
    def stride_aligned(cls, value: int) -> int:
        if value % 32:
            raise ValueError("image_size must be a multiple of 32")
        return value


class ClassificationJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: UUID
    model_arch: Literal["resnet18"]
    hyperparams: ClassificationParameters = Field(
        default_factory=ClassificationParameters
    )


class DetectionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: UUID
    model_arch: Literal["yolov8s"]
    hyperparams: DetectionParameters = Field(default_factory=DetectionParameters)


JobCreate = Annotated[
    ClassificationJob | DetectionJob, Field(discriminator="model_arch")
]


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    dataset_version_id: UUID
    model_arch: str
    hyperparams: dict[str, Any]
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    gpu_id: int | None
    dispatched_at: datetime | None
    cancel_requested_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class JobPage(BaseModel):
    items: list[JobResponse]
    total: int
    page: int
    page_size: int


class MetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epoch: int
    split: str
    name: str
    value: float


class MetricPage(BaseModel):
    items: list[MetricResponse]
    total: int
    page: int
    page_size: int
