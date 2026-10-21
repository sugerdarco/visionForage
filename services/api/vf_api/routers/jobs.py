"""Owner-scoped training requests; PostgreSQL is the durable queue outbox."""

from datetime import datetime, timezone
import re
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from vf_api.dependencies import CurrentUser, Database, OwnedProject
from vf_api.job_schemas import JobCreate, JobPage, JobResponse, MetricPage
from vf_api.models import DatasetVersion, Model, Project, TrainingJob, TrainingMetric

router = APIRouter(tags=["training jobs"])
Page = Annotated[int, Query(ge=1, le=1_000_000)]
PageSize = Annotated[int, Query(ge=1, le=100)]


def job_responses(rows, session):
    rows = list(rows)
    identifiers = [row.id for row in rows if row.status == "succeeded"]
    models = (
        dict(
            session.execute(
                select(Model.job_id, Model.id).where(Model.job_id.in_(identifiers))
            ).all()
        )
        if identifiers
        else {}
    )
    return [
        JobResponse.model_validate(row).model_copy(
            update={"model_id": models.get(row.id)}
        )
        for row in rows
    ]


def owned_job(job_id, session, user, *, lock=False):
    statement = (
        select(TrainingJob)
        .join(Project, Project.id == TrainingJob.project_id)
        .where(TrainingJob.id == job_id, Project.owner_id == user.id)
    )
    if lock:
        statement = statement.with_for_update(of=TrainingJob)
    row = session.scalar(statement)
    if row is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return row


@router.post("/projects/{project_id}/jobs", response_model=JobResponse, status_code=201)
def create_job(body: JobCreate, project: OwnedProject, session: Database):
    version = session.scalar(
        select(DatasetVersion).where(
            DatasetVersion.id == body.dataset_version_id,
            DatasetVersion.project_id == project.id,
        )
    )
    if version is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    task = {"resnet18": "classification", "yolov8s": "detection"}[body.model_arch]
    if project.task_type != task:
        raise HTTPException(
            status_code=422, detail="Model does not match the project task"
        )
    counts = version.counts
    hashes = (counts.get("manifest_sha256"), counts.get("audit_sha256"))
    if not all(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
        for value in hashes
    ) or not counts.get("audit_key"):
        raise HTTPException(
            status_code=422,
            detail="Create a checksummed dataset version before training",
        )
    images = counts.get("images", {})
    if not isinstance(images, dict) or any(
        type(images.get(split)) is not int or images[split] < minimum
        for split, minimum in (("train", 1), ("val", 1), ("test", 32))
    ):
        raise HTTPException(
            status_code=422,
            detail="Training needs nonempty train/val splits and at least 32 test images for ONNX verification",
        )
    row = TrainingJob(
        project_id=project.id,
        dataset_version_id=version.id,
        model_arch=body.model_arch,
        hyperparams=body.hyperparams.model_dump(),
        celery_task_id=str(uuid4()),
    )
    session.add(row)
    # No Redis call here: a committed queued row is itself a durable dispatch request.
    session.commit()
    return row


@router.get("/projects/{project_id}/jobs", response_model=JobPage)
def list_jobs(
    project: OwnedProject, session: Database, page: Page = 1, page_size: PageSize = 20
):
    condition = TrainingJob.project_id == project.id
    total = session.scalar(
        select(func.count()).select_from(TrainingJob).where(condition)
    )
    rows = session.scalars(
        select(TrainingJob)
        .where(condition)
        .order_by(TrainingJob.created_at.desc(), TrainingJob.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return JobPage(
        items=job_responses(rows, session), total=total, page=page, page_size=page_size
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def read_job(job_id: UUID, session: Database, user: CurrentUser):
    return job_responses([owned_job(job_id, session, user)], session)[0]


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: UUID, session: Database, user: CurrentUser):
    row = owned_job(job_id, session, user, lock=True)
    if row.status in ("queued", "running") and row.cancel_requested_at is None:
        row.cancel_requested_at = datetime.now(timezone.utc)
        if row.status == "queued":
            row.status = "cancelled"
            row.finished_at = row.cancel_requested_at
        # A running worker must acknowledge the request before it becomes terminal.
        session.commit()
    return job_responses([row], session)[0]


@router.get("/jobs/{job_id}/metrics", response_model=MetricPage)
def read_metrics(
    job_id: UUID,
    session: Database,
    user: CurrentUser,
    page: Page = 1,
    page_size: PageSize = 100,
):
    owned_job(job_id, session, user)
    condition = TrainingMetric.job_id == job_id
    total = session.scalar(
        select(func.count()).select_from(TrainingMetric).where(condition)
    )
    rows = session.scalars(
        select(TrainingMetric)
        .where(condition)
        .order_by(TrainingMetric.epoch, TrainingMetric.split, TrainingMetric.name)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return MetricPage(items=list(rows), total=total, page=page, page_size=page_size)
