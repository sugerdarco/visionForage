"""Validated persisted predictions and owner-facing monitoring responses."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]


class CheckedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ClassificationOutput(CheckedModel):
    label: str
    confidence: Probability
    scores: dict[str, Probability] = Field(min_length=1, max_length=1000)


class PredictionBox(CheckedModel):
    label: str
    confidence: Probability
    x: Probability
    y: Probability
    w: Annotated[float, Field(gt=0, le=1)]
    h: Annotated[float, Field(gt=0, le=1)]

    @model_validator(mode="after")
    def fits_image(self):
        if self.x + self.w > 1.000001 or self.y + self.h > 1.000001:
            raise ValueError("Box exceeds image")
        return self


class DetectionOutput(CheckedModel):
    boxes: list[PredictionBox] = Field(max_length=300)


class PredictionEvent(CheckedModel):
    schema_version: Literal[1]
    prediction_id: UUID
    deployment_id: UUID
    model_id: UUID
    model_version: int = Field(ge=1)
    created_at: AwareDatetime
    latency_ms: float = Field(ge=0)
    mime: Literal["image/jpeg", "image/png"]
    image_bytes: int = Field(gt=0, le=20 * 1024 * 1024)
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output: ClassificationOutput | DetectionOutput


class PredictionView(BaseModel):
    id: UUID
    deployment_id: UUID
    model_id: UUID
    model_version: int
    output: ClassificationOutput | DetectionOutput
    max_confidence: float
    latency_ms: float
    sent_to_review: bool
    created_at: datetime
    image_url: str
    image_url_expires_in: int = 300


class PredictionPage(BaseModel):
    items: list[PredictionView]
    total: int
    page: int
    page_size: int


class ConfidenceBin(BaseModel):
    lower: float
    upper: float
    upper_inclusive: bool
    count: int


class DeploymentStats(BaseModel):
    deployment_id: UUID
    window: Literal["1h", "24h", "7d", "30d"]
    start: datetime
    end: datetime
    scope: Literal["recorded_predictions"] = "recorded_predictions"
    requests: int
    low_confidence_count: int
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    class_distribution: dict[str, int]
    class_count_unit: Literal["predictions", "boxes"]
    confidence_histogram: list[ConfidenceBin]
