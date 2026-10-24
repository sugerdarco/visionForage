"""Atomic, owner-scoped replacement of an image's live labels."""

from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, select

from vf_api.annotation_schemas import (
    AnnotationResponse,
    AnnotationSave,
    ImageAnnotations,
)
from vf_api.dependencies import CurrentUser, Database, owned_project
from vf_api.models import Annotation, Class, Image, Project

router = APIRouter(prefix="/images", tags=["annotations"])


def validate_labels(body, task, project_id, session):
    if task not in ("classification", "detection"):
        raise HTTPException(status_code=422, detail="Task does not support labeling")

    status = body.status or ("labeled" if body.annotations else "unlabeled")
    if status == "unlabeled" and body.annotations:
        raise HTTPException(
            status_code=422, detail="Unlabeled images cannot contain annotations"
        )
    if task == "classification":
        if status == "labeled" and len(body.annotations) != 1:
            raise HTTPException(
                status_code=422, detail="Classification requires exactly one class"
            )
        if any(row.bbox is not None for row in body.annotations):
            raise HTTPException(
                status_code=422, detail="Classification labels cannot contain boxes"
            )
    elif any(row.bbox is None for row in body.annotations):
        raise HTTPException(
            status_code=422, detail="Detection labels require bounding boxes"
        )

    requested_ids = {row.class_id for row in body.annotations}
    valid_ids = set(
        session.scalars(
            select(Class.id).where(
                Class.project_id == project_id, Class.id.in_(requested_ids)
            )
        )
    )
    if requested_ids != valid_ids:
        raise HTTPException(
            status_code=422, detail="Every class must belong to the image's project"
        )

    return status


@router.get("/{image_id}/annotations", response_model=ImageAnnotations)
def read_annotations(
    image_id: UUID, session: Database, user: CurrentUser
) -> ImageAnnotations:
    # One statement gives a consistent image status and label snapshot during saves.
    rows = session.execute(
        select(Image, Annotation)
        .join(Project, Project.id == Image.project_id)
        .outerjoin(Annotation, Annotation.image_id == Image.id)
        .where(Image.id == image_id, Project.owner_id == user.id)
        .order_by(Annotation.id)
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageAnnotations(
        image_id=image_id,
        status=rows[0][0].status,
        annotations=[row for _, row in rows if row is not None],
    )


@router.put("/{image_id}/annotations", response_model=ImageAnnotations)
def save_annotations(
    image_id: UUID, body: AnnotationSave, session: Database, user: CurrentUser
) -> ImageAnnotations:
    project_id = session.scalar(
        select(Image.project_id)
        .join(Project)
        .where(Image.id == image_id, Project.owner_id == user.id)
    )
    if project_id is None:
        raise HTTPException(status_code=404, detail="Image not found")
    owned_project(project_id, session, user, lock=True)
    result = session.execute(
        select(Image, Project.task_type)
        .join(Project, Project.id == Image.project_id)
        .where(Image.id == image_id, Project.owner_id == user.id)
        .with_for_update(of=Image)
    ).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found")
    image, task = result
    status = validate_labels(body, task, image.project_id, session)

    existing = list(
        session.scalars(select(Annotation).where(Annotation.image_id == image.id))
    )

    def label_key(row):
        return str(row.class_id), tuple(row.bbox or ())

    unchanged = all(
        row.source == "human" and row.confidence is None for row in existing
    ) and sorted(map(label_key, existing)) == sorted(map(label_key, body.annotations))
    if not unchanged:
        session.execute(delete(Annotation).where(Annotation.image_id == image.id))
        existing = [
            Annotation(
                image_id=image.id,
                class_id=row.class_id,
                bbox=list(row.bbox) if row.bbox is not None else None,
                source="human",
                confidence=None,
            )
            for row in body.annotations
        ]
        session.add_all(existing)
    image.status = status
    session.flush()
    # Materialize under the lock; another save can begin as soon as we commit.
    response = ImageAnnotations(
        image_id=image.id,
        status=image.status,
        annotations=[
            AnnotationResponse.model_validate(row)
            for row in sorted(existing, key=lambda row: row.id)
        ],
    )
    session.commit()
    return response
