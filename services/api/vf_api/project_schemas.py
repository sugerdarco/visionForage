"""Project inputs and public class maps, separate from account credentials."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


def valid_name(value: str) -> str:
    if "\0" in value:
        raise ValueError("Names cannot contain null characters")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("Names must be valid UTF-8") from None
    return value


Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    AfterValidator(valid_name),
]


class ClassCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Name
    color: Annotated[
        str,
        StringConstraints(pattern=r"^#[0-9a-fA-F]{6}$", max_length=7),
        AfterValidator(str.lower),
    ]


class ClassBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ClassCreate] = Field(min_length=1, max_length=100)

    @field_validator("classes")
    @classmethod
    def unique_names(cls, values: list[ClassCreate]) -> list[ClassCreate]:
        if len({value.name for value in values}) != len(values):
            raise ValueError("Class names must be unique within a batch")
        return values


class ProjectCreate(ClassBatch):
    name: Name
    task_type: Literal["classification", "detection"]
    classes: list[ClassCreate] = Field(default_factory=list, max_length=100)


class ProjectRename(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Name


class ClassResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    name: str
    color: str
    index: int


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: UUID
    name: str
    task_type: Literal["classification", "detection", "anomaly"]
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectResponse):
    classes: list[ClassResponse]


class ProjectPage(BaseModel):
    items: list[ProjectResponse]
    page: int
    page_size: int
    total: int
