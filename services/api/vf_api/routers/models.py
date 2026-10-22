"""Owner-scoped browsing of models already registered by the training worker."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select, update

from vf_api.dependencies import CurrentUser, Database, OwnedProject, owned_project
from vf_api.model_schemas import ModelDetail, ModelPage
from vf_api.models import DatasetVersion, Model, Project, TrainingJob

router = APIRouter(tags=["model registry"])
Page = Annotated[int, Query(ge=1, le=1_000_000)]
PageSize = Annotated[int, Query(ge=1, le=100)]


def registry_query(*, detail=False):
    # Select only public fields: metrics can contain large private artifact indexes.
    fields = [
        Model.id,
        Model.project_id,
        Model.job_id,
        Model.version,
        Project.task_type,
        TrainingJob.model_arch,
        TrainingJob.dataset_version_id,
        DatasetVersion.version.label("dataset_version"),
        Model.input_size,
        Model.is_promoted,
        Model.created_at,
        Model.updated_at,
    ]
    if detail:
        fields.extend([Model.class_map, TrainingJob.hyperparams])
    return (
        select(*fields)
        .select_from(Model)
        .join(Project, Project.id == Model.project_id)
        .join(TrainingJob, TrainingJob.id == Model.job_id)
        .join(DatasetVersion, DatasetVersion.id == TrainingJob.dataset_version_id)
    )


@router.get("/projects/{project_id}/models", response_model=ModelPage)
def list_models(
    project: OwnedProject, session: Database, page: Page = 1, page_size: PageSize = 20
):
    condition = Model.project_id == project.id
    total = session.scalar(select(func.count()).select_from(Model).where(condition))
    rows = session.execute(
        registry_query()
        .where(condition)
        .order_by(Model.version.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).mappings()
    return ModelPage(items=list(rows), total=total, page=page, page_size=page_size)


@router.get("/models/{model_id}", response_model=ModelDetail)
def read_model(model_id: UUID, session: Database, user: CurrentUser):
    row = (
        session.execute(
            registry_query(detail=True).where(
                Model.id == model_id, Project.owner_id == user.id
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelDetail.model_validate(row)


@router.post("/models/{model_id}/promote", response_model=ModelDetail)
def promote_model(model_id: UUID, session: Database, user: CurrentUser):
    target = read_model(model_id, session, user)
    # Registration uses this same project lock. Always acquire it before changing
    # model rows so simultaneous promotions serialize without a uniqueness race.
    owned_project(target.project_id, session, user, lock=True)
    target = read_model(model_id, session, user)
    status = session.scalar(
        select(TrainingJob.status).where(TrainingJob.id == target.job_id)
    )
    if status != "succeeded":
        raise HTTPException(status_code=409, detail="Model training has not succeeded")
    if not target.is_promoted:
        session.execute(
            update(Model)
            .where(Model.project_id == target.project_id, Model.is_promoted.is_(True))
            .values(is_promoted=False)
        )
        session.execute(
            update(Model).where(Model.id == model_id).values(is_promoted=True)
        )
    result = read_model(model_id, session, user)
    session.commit()
    return result
