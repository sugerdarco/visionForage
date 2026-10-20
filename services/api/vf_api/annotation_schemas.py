"""Human label edits and normalized top-left bounding boxes."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

UnitCoordinate = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
BoxExtent = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
BoundingBox = tuple[UnitCoordinate, UnitCoordinate, BoxExtent, BoxExtent]


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: UUID
    bbox: BoundingBox | None = None

    @field_validator("bbox")
    @classmethod
    def within_image(cls, box: BoundingBox | None) -> BoundingBox | None:
        if box is not None:
            x, y, width, height = box
            # Match the database's tolerance for floating-point edge arithmetic.
            if x + width > 1.000000001 or y + height > 1.000000001:
                raise ValueError("Bounding box must fit inside the image")
        return box


class AnnotationSave(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotations: list[AnnotationInput] = Field(max_length=1000)
    # Explicit labeled + [] represents a reviewed negative detection image.
    status: Literal["labeled", "unlabeled"] | None = None


class AnnotationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    class_id: UUID
    bbox: BoundingBox | None
    source: Literal["human", "model"]
    confidence: float | None


class ImageAnnotations(BaseModel):
    image_id: UUID
    status: Literal["unlabeled", "labeled", "in_review"]
    annotations: list[AnnotationResponse]
