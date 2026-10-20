"""Owner-scoped projects and append-only model class indices."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vf_api.dependencies import CurrentUser, Database, LockedProject, OwnedProject
from vf_api.models import Class, Project
from vf_api.project_schemas import (
    ClassBatch,
    ClassCreate,
    ClassResponse,
    ProjectCreate,
    ProjectDetail,
    ProjectPage,
    ProjectRename,
    ProjectResponse,
)

router = APIRouter(prefix="/projects", tags=["projects"])


def class_rows(project: Project, session: Session) -> list[Class]:
    return list(
        session.scalars(
            select(Class).where(Class.project_id == project.id).order_by(Class.index)
        )
    )


def detail(project: Project, session: Session) -> ProjectDetail:
    return ProjectDetail(
        **ProjectResponse.model_validate(project).model_dump(),
        classes=class_rows(project, session),
    )


def append_classes(
    project: Project, values: list[ClassCreate], session: Session
) -> list[Class]:
    # Callers hold the project row lock, or have just inserted a private new row.
    last_index = session.scalar(
        select(func.max(Class.index)).where(Class.project_id == project.id)
    )
    first_index = 0 if last_index is None else last_index + 1
    rows = [
        Class(project_id=project.id, index=first_index + offset, **value.model_dump())
        for offset, value in enumerate(values)
    ]
    session.add_all(rows)
    return rows


def commit_classes(session: Session) -> None:
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        constraint = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
        if constraint == "uq_classes_project_name":
            raise HTTPException(
                status_code=409, detail="Class name already exists"
            ) from None
        raise


@router.post("", response_model=ProjectDetail, status_code=201)
def create_project(
    body: ProjectCreate, session: Database, user: CurrentUser
) -> ProjectDetail:
    project = Project(owner_id=user.id, name=body.name, task_type=body.task_type)
    session.add(project)
    session.flush()
    append_classes(project, body.classes, session)
    commit_classes(session)
    return detail(project, session)


@router.get("", response_model=ProjectPage)
def list_projects(
    session: Database,
    user: CurrentUser,
    page: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ProjectPage:
    owned = Project.owner_id == user.id
    total = session.scalar(select(func.count()).select_from(Project).where(owned))
    rows = session.scalars(
        select(Project)
        .where(owned)
        .order_by(Project.created_at.desc(), Project.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return ProjectPage(items=list(rows), total=total, page=page, page_size=page_size)


@router.get("/{project_id}", response_model=ProjectDetail)
def read_project(project: OwnedProject, session: Database) -> ProjectDetail:
    return detail(project, session)


@router.patch("/{project_id}", response_model=ProjectDetail)
def rename_project(
    body: ProjectRename, project: LockedProject, session: Database
) -> ProjectDetail:
    project.name = body.name
    session.commit()
    session.refresh(project)
    return detail(project, session)


@router.delete("/{project_id}", status_code=204)
def delete_project(project: LockedProject, session: Database) -> Response:
    try:
        session.execute(delete(Class).where(Class.project_id == project.id))
        session.delete(project)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        if getattr(error.orig, "sqlstate", None) == "23503":
            raise HTTPException(
                status_code=409,
                detail="Project contains data; deletion requires a cleanup workflow",
            ) from None
        raise
    return Response(status_code=204)


@router.get("/{project_id}/classes", response_model=list[ClassResponse])
def list_classes(project: OwnedProject, session: Database) -> list[Class]:
    return class_rows(project, session)


@router.post(
    "/{project_id}/classes", response_model=list[ClassResponse], status_code=201
)
def add_classes(
    body: ClassBatch, project: LockedProject, session: Database
) -> list[Class]:
    rows = append_classes(project, body.classes, session)
    commit_classes(session)
    return rows
