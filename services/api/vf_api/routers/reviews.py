"""Turn corrected low-confidence predictions into labeled project images."""

from datetime import datetime
import hashlib
import json
from typing import Annotated
from uuid import UUID
from botocore.exceptions import ClientError

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from vf_api.annotation_schemas import AnnotationInput, AnnotationSave
from vf_api.dependencies import CurrentUser, Database, OwnedProject, owned_project
from vf_api.image_files import decode_image
from vf_api.models import Annotation, Deployment, Image, Model, Prediction, Project
from vf_api.prediction_schemas import PredictionPage, PredictionView
from vf_api.routers.annotations import validate_labels
from vf_api.routers.uploads import request_thumbnail
from vf_api.upload_dependencies import Predictions, Queue, Storage

router = APIRouter(tags=["review"])


class ReviewAccept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotations: list[AnnotationInput] = Field(max_length=1000)


class ReviewReceipt(BaseModel):
    prediction_id: UUID
    image_id: UUID
    reviewed_at: datetime


@router.get("/projects/{project_id}/review-queue", response_model=PredictionPage)
def review_queue(
    project: OwnedProject,
    session: Database,
    store: Predictions,
    page: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    query = (
        select(Prediction, Model)
        .join(Deployment, Deployment.id == Prediction.deployment_id)
        .join(Model, Model.id == Deployment.model_id)
        .where(
            Model.project_id == project.id,
            Prediction.sent_to_review.is_(True),
            Prediction.reviewed_at.is_(None),
        )
    )
    total = session.scalar(select(func.count()).select_from(query.subquery()))
    rows = session.execute(
        query.order_by(Prediction.max_confidence, Prediction.created_at, Prediction.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return PredictionPage(
        items=[
            PredictionView(
                id=row.id,
                deployment_id=row.deployment_id,
                model_id=model.id,
                model_version=model.version,
                output=row.output,
                max_confidence=row.max_confidence,
                latency_ms=row.latency_ms,
                sent_to_review=row.sent_to_review,
                created_at=row.created_at,
                image_url=store.url(row.image_key),
            )
            for row, model in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/review/{prediction_id}/accept", response_model=ReviewReceipt)
def accept_review(
    prediction_id: UUID,
    body: ReviewAccept,
    session: Database,
    user: CurrentUser,
    predictions: Predictions,
    images: Storage,
    queue: Queue,
):
    project_id = session.scalar(
        select(Model.project_id)
        .select_from(Prediction)
        .join(Deployment, Deployment.id == Prediction.deployment_id)
        .join(Model, Model.id == Deployment.model_id)
        .join(Project, Project.id == Model.project_id)
        .where(Prediction.id == prediction_id, Project.owner_id == user.id)
    )
    if project_id is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    project = owned_project(project_id, session, user, lock=True)
    prediction = session.scalar(
        select(Prediction).where(Prediction.id == prediction_id).with_for_update()
    )
    validate_labels(
        AnnotationSave(annotations=body.annotations, status="labeled"),
        project.task_type,
        project_id,
        session,
    )
    labels = sorted(
        (str(label.class_id), tuple(label.bbox or ())) for label in body.annotations
    )
    digest = hashlib.sha256(
        json.dumps(labels, separators=(",", ":")).encode()
    ).hexdigest()
    if prediction.reviewed_at is not None:
        if prediction.review_payload_sha256 != digest:
            raise HTTPException(
                status_code=409,
                detail="Prediction was already accepted with different labels",
            )
        return ReviewReceipt(
            prediction_id=prediction_id,
            image_id=prediction.review_image_id,
            reviewed_at=prediction.reviewed_at,
        )
    if not prediction.sent_to_review:
        raise HTTPException(status_code=409, detail="Prediction is not awaiting review")
    if not prediction.image_key.startswith(f"{prediction.deployment_id}/"):
        raise HTTPException(status_code=409, detail="Prediction image unavailable")
    try:
        data = predictions.read(prediction.image_key)
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            raise HTTPException(
                status_code=409, detail="Prediction image unavailable"
            ) from None
        raise
    try:
        with decode_image(data) as decoded:
            width, height = decoded.size
    except ValueError:
        raise HTTPException(
            status_code=409, detail="Prediction image unavailable"
        ) from None
    sha = hashlib.sha256(data).hexdigest()
    image = session.scalar(
        select(Image)
        .where(Image.project_id == project_id, Image.sha256 == sha)
        .with_for_update()
    )
    if image is not None:
        existing = list(
            session.scalars(select(Annotation).where(Annotation.image_id == image.id))
        )
        current = sorted((str(row.class_id), tuple(row.bbox or ())) for row in existing)
        if not (image.status == "unlabeled" and not existing) and not (
            image.status == "labeled"
            and current == labels
            and all(
                row.source == "human" and row.confidence is None for row in existing
            )
        ):
            raise HTTPException(
                status_code=409, detail="Existing project image has conflicting labels"
            )
    else:
        suffix = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpg"
        mime = "image/png" if suffix == "png" else "image/jpeg"
        key = f"{project_id}/originals/{sha}.{suffix}"
        images.put(key, data, mime)
        image = Image(
            project_id=project_id,
            storage_key=key,
            width=width,
            height=height,
            sha256=sha,
            status="unlabeled",
        )
        session.add(image)
        session.flush()
    if image.status == "unlabeled":
        image.status = "labeled"
        session.add_all(
            Annotation(
                image_id=image.id,
                class_id=row.class_id,
                bbox=list(row.bbox) if row.bbox else None,
                source="human",
                confidence=None,
            )
            for row in body.annotations
        )
    prediction.review_image_id = image.id
    prediction.review_payload_sha256 = digest
    prediction.reviewed_at = session.scalar(select(func.clock_timestamp()))
    receipt = ReviewReceipt(
        prediction_id=prediction_id,
        image_id=image.id,
        reviewed_at=prediction.reviewed_at,
    )
    session.commit()
    request_thumbnail(image, queue)
    return receipt
