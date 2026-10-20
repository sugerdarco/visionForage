"""Dataset snapshot options and owner-visible version records."""

from datetime import datetime
import math
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitRatios(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: float = Field(default=0.7, strict=True, gt=0, lt=1, allow_inf_nan=False)
    val: float = Field(default=0.15, strict=True, gt=0, lt=1, allow_inf_nan=False)
    test: float = Field(default=0.15, strict=True, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def sum_to_one(self):
        if not math.isclose(
            self.train + self.val + self.test, 1, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("Split ratios must sum to one")
        return self


class VersionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, strict=True, ge=0, le=2**32 - 1)
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)


class VersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    version: int
    manifest_key: str
    split_ratios: dict[str, float]
    counts: dict[str, Any]
    created_at: datetime
    manifest_url: str
    audit_url: str | None
    urls_expire_in: int


class VersionPage(BaseModel):
    items: list[VersionResponse]
    page: int
    page_size: int
    total: int
