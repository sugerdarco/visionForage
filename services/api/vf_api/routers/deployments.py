"""Synchronous local deployments with durable reservations and serialized operations."""

import hashlib
import secrets
from typing import Annotated
from uuid import UUID, uuid4

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from vf_api.deployment_manager import DeploymentManager
from vf_api.deployment_runtime import DeploymentUnavailable
from vf_api.deployment_schemas import (
    DeploymentCreate,
    DeploymentCreated,
    DeploymentPage,
    DeploymentResponse,
)
from vf_api.dependencies import CurrentUser, Database, OwnedProject
from vf_api.model_artifacts import artifact_record
from vf_api.models import Deployment, Model, Project, TrainingJob
from vf_api.routers.models import owned_model
from vf_api.settings import DeploymentSettings
from vf_api.upload_dependencies import Artifacts

router = APIRouter(tags=["deployments"])


def get_manager(request: Request):
    with request.app.state.resource_lock:
        if not hasattr(request.app.state, "deployments"):
            request.app.state.deployments = DeploymentManager(DeploymentSettings())
    return request.app.state.deployments


Manager = Annotated[DeploymentManager, Depends(get_manager)]


def operation_lock(session, identifier, *, wait=False):
    if wait:
        session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"visionforge-deployment:{identifier}"},
        )
        return True
    return session.scalar(
        text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"visionforge-deployment:{identifier}"},
    )


def view(row, project_id, model_version, *, endpoint=None, checked=False):
    return DeploymentResponse(
        id=row.id,
        model_id=row.model_id,
        project_id=project_id,
        model_version=model_version,
        slug=row.slug,
        status=row.status,
        confidence_threshold=row.confidence_threshold,
        endpoint_url=endpoint,
        runtime_checked=checked,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def owned_deployment(identifier, session, user):
    result = session.execute(
        select(Deployment, Model.project_id, Model.version)
        .join(Model, Model.id == Deployment.model_id)
        .join(Project, Project.id == Model.project_id)
        .where(Deployment.id == identifier, Project.owner_id == user.id)
    ).one_or_none()
    if result is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return result


def identity(row, project_id):
    return {"id": row.id, "model_id": row.model_id, "project_id": project_id}


@router.post(
    "/models/{model_id}/deployments", response_model=DeploymentCreated, status_code=201
)
def create_deployment(
    model_id: UUID,
    body: DeploymentCreate,
    session: Database,
    user: CurrentUser,
    manager: Manager,
    store: Artifacts,
):
    model = owned_model(model_id, session, user)
    if (
        session.scalar(select(TrainingJob.status).where(TrainingJob.id == model.job_id))
        != "succeeded"
    ):
        raise HTTPException(status_code=409, detail="Model training has not succeeded")
    project_id, version = model.project_id, model.version
    graph, metadata = (
        artifact_record(model, "onnx/model.onnx"),
        artifact_record(model, "onnx/metadata.json"),
    )
    if metadata["bytes"] > 1024**2:
        raise HTTPException(
            status_code=409, detail="Serving metadata exceeds the supported limit"
        )
    image = manager.runtime.image()
    # Serialize only capacity reservations, not artifact downloads or startup.
    if not operation_lock(session, "capacity"):
        raise HTTPException(
            status_code=409,
            detail="Another deployment is being reserved; retry shortly",
        )
    active = session.scalar(
        select(func.count())
        .select_from(Deployment)
        .where(Deployment.status.in_(["starting", "running"]))
    )
    if active >= manager.settings.deployment_max_active:
        raise HTTPException(status_code=409, detail="Active deployment limit reached")
    key = secrets.token_urlsafe(32)
    identifier = uuid4()
    row = Deployment(
        id=identifier,
        model_id=model.id,
        slug=body.slug or f"model-{identifier.hex}",
        api_key_hash=hashlib.sha256(key.encode()).hexdigest(),
        confidence_threshold=body.confidence_threshold,
    )
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409, detail="Deployment slug is already used"
        ) from None
    operation_lock(session, identifier, wait=True)
    session.refresh(row)
    if row.status != "starting":
        raise HTTPException(status_code=409, detail="Deployment is no longer starting")
    spec = {
        **identity(row, project_id),
        "slug": row.slug,
        "model_version": version,
        "key_hash": row.api_key_hash,
        "model": graph,
        "metadata": metadata,
    }

    def check():
        # A lost DB connection releases the operation lock. Do not continue to
        # another Docker side effect after losing that ownership connection.
        if (
            session.scalar(select(Deployment.status).where(Deployment.id == identifier))
            != "starting"
        ):
            raise DeploymentUnavailable("Deployment ownership changed")

    endpoint = None
    try:
        info = manager.launch(spec, store, image, check)
        check()
        endpoint = manager.runtime.endpoint(info, row.slug)
        row.container_id, row.status = info["Id"], "running"
    except (DeploymentUnavailable, BotoCoreError, ClientError, OSError, ValueError):
        # Only claim a terminal failure after removal is confirmed. If Docker is
        # unreachable, leave the durable starting reservation for reconciliation.
        manager.cleanup(spec)
        row.container_id, row.status = None, "failed"
    session.commit()
    return DeploymentCreated(
        **view(row, project_id, version, endpoint=endpoint, checked=True).model_dump(),
        api_key=key if row.status == "running" else None,
    )


@router.get("/projects/{project_id}/deployments", response_model=DeploymentPage)
def list_deployments(
    project: OwnedProject,
    session: Database,
    page: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    condition = Model.project_id == project.id
    total = session.scalar(
        select(func.count()).select_from(Deployment).join(Model).where(condition)
    )
    rows = session.execute(
        select(Deployment, Model.version)
        .join(Model)
        .where(condition)
        .order_by(Deployment.created_at.desc(), Deployment.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return DeploymentPage(
        items=[view(row, project.id, version) for row, version in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
def read_deployment(
    deployment_id: UUID, session: Database, user: CurrentUser, manager: Manager
):
    row, project, version = owned_deployment(deployment_id, session, user)
    if not operation_lock(session, deployment_id):
        return view(row, project, version)
    session.refresh(row)
    spec = identity(row, project)
    info = manager.runtime.inspect(spec)
    endpoint = None
    if row.status in ("starting", "running") and manager.runtime.healthy(info):
        endpoint = manager.runtime.endpoint(info, row.slug)
        row.container_id, row.status = info["Id"], "running"
    elif (
        row.status == "starting"
        and (
            session.scalar(select(func.clock_timestamp())) - row.created_at
        ).total_seconds()
        < manager.settings.deployment_ready_seconds + 30
    ):
        # The reservation commits just before the creator acquires its operation
        # lock. A read in that gap must not fail a launch that is about to begin.
        return view(row, project, version, checked=True)
    else:
        manager.cleanup(spec)
        row.container_id = None
        if row.status != "stopped":
            row.status = "failed"
    session.commit()
    return view(row, project, version, endpoint=endpoint, checked=True)


@router.delete("/deployments/{deployment_id}", response_model=DeploymentResponse)
def stop_deployment(
    deployment_id: UUID, session: Database, user: CurrentUser, manager: Manager
):
    row, project, version = owned_deployment(deployment_id, session, user)
    if not operation_lock(session, deployment_id):
        raise HTTPException(
            status_code=409, detail="Deployment operation in progress; retry shortly"
        )
    session.refresh(row)
    manager.cleanup(identity(row, project))
    row.container_id, row.status = None, "stopped"
    session.commit()
    return view(row, project, version, checked=True)
