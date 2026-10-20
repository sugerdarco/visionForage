"""Freeze labels into private, create-only training artifacts."""

from threading import BoundedSemaphore
from typing import Annotated
from uuid import UUID, uuid4

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from vf_api.dataset_snapshots import SnapshotError, build_snapshot, copy_snapshot
from vf_api.dependencies import CurrentUser, Database, OwnedProject, owned_project
from vf_api.models import DatasetVersion
from vf_api.storage import DatasetStore, READ_SECONDS
from vf_api.upload_dependencies import Datasets, Storage
from vf_api.version_schemas import VersionCreate, VersionPage, VersionResponse

router = APIRouter(
    prefix="/projects/{project_id}/dataset-versions", tags=["dataset versions"]
)
slots = BoundedSemaphore(1)


def version_response(row: DatasetVersion, datasets: DatasetStore) -> VersionResponse:
    return VersionResponse(
        id=row.id,
        project_id=row.project_id,
        version=row.version,
        manifest_key=row.manifest_key,
        split_ratios=row.split_ratios,
        counts=row.counts,
        created_at=row.created_at,
        manifest_url=datasets.url(row.manifest_key),
        audit_url=datasets.url(row.counts["audit_key"])
        if "audit_key" in row.counts
        else None,
        urls_expire_in=READ_SECONDS,
    )


@router.post("", response_model=VersionResponse, status_code=201)
def create_version(
    body: VersionCreate,
    project: OwnedProject,
    user: CurrentUser,
    session: Database,
    store: Storage,
    datasets: Datasets,
) -> VersionResponse:
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Another version build is running; retry shortly",
            headers={"Retry-After": "5"},
        )
    try:
        project_id = project.id
        owned_project(project_id, session, user, lock=True)
        snapshot = copy_snapshot(project, session)
        # All values above are copied. Release editing locks before slow object I/O.
        session.rollback()
        manifest, audit, counts = build_snapshot(snapshot, body, store)
        owned_project(project_id, session, user, lock=True)
        last = session.scalar(
            select(func.max(DatasetVersion.version)).where(
                DatasetVersion.project_id == project_id
            )
        )
        number = (last or 0) + 1
        prefix = f"{project_id}/v{number}/{uuid4()}"
        manifest_key, audit_key = f"{prefix}/manifest.json", f"{prefix}/audit.json"
        # Unique attempt namespace avoids colliding with objects from rolled-back builds.
        datasets.put_once(manifest_key, manifest)
        datasets.put_once(audit_key, audit)
        counts["audit_key"] = audit_key
        row = DatasetVersion(
            project_id=project_id,
            version=number,
            manifest_key=manifest_key,
            split_ratios=body.split_ratios.model_dump(),
            counts=counts,
        )
        session.add(row)
        session.commit()
        return version_response(row, datasets)
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=409,
                detail="A registered original is missing; restore it before building a version",
            ) from None
        raise
    except SnapshotError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except ValueError:
        raise HTTPException(
            status_code=422, detail="Invalid image or label in the snapshot"
        ) from None
    finally:
        slots.release()


@router.get("", response_model=VersionPage)
def list_versions(
    project: OwnedProject,
    session: Database,
    datasets: Datasets,
    page: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> VersionPage:
    condition = DatasetVersion.project_id == project.id
    total = session.scalar(
        select(func.count()).select_from(DatasetVersion).where(condition)
    )
    rows = session.scalars(
        select(DatasetVersion)
        .where(condition)
        .order_by(DatasetVersion.version.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return VersionPage(
        items=[version_response(row, datasets) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{version_id}", response_model=VersionResponse)
def read_version(
    version_id: UUID, project: OwnedProject, session: Database, datasets: Datasets
) -> VersionResponse:
    row = session.scalar(
        select(DatasetVersion).where(
            DatasetVersion.id == version_id, DatasetVersion.project_id == project.id
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    return version_response(row, datasets)
