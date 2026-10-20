"""Direct archive uploads followed by bounded local CPU validation and import."""

from tempfile import TemporaryFile
from threading import BoundedSemaphore
from zipfile import BadZipFile, LargeZipFile
import zlib

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
import jwt
import yaml

from vf_api.dataset_imports import register_dataset
from vf_api.dependencies import Configuration, CurrentUser, Database, OwnedProject
from vf_api.import_archive import Archive
from vf_api.import_formats import parse_dataset
from vf_api.import_schemas import ImportClaim, ImportResponse, InvalidDataset
from vf_api.import_tokens import grant_import, read_import
from vf_api.storage import UPLOAD_SECONDS
from vf_api.upload_dependencies import Queue, Storage
from vf_api.upload_schemas import CompleteRequest, UploadPermission

router = APIRouter(prefix="/projects/{project_id}/imports", tags=["imports"])
# Each API process validates one archive at a time; other callers can retry.
slots = BoundedSemaphore(1)


def check_task(claim: ImportClaim, task: str) -> None:
    expected = "classification" if claim.format == "imagefolder" else "detection"
    if task != expected:
        raise HTTPException(
            status_code=422, detail="Import format does not match project task"
        )


@router.post("/presign", response_model=UploadPermission)
def presign_import(
    body: ImportClaim,
    project: OwnedProject,
    user: CurrentUser,
    settings: Configuration,
    store: Storage,
) -> UploadPermission:
    check_task(body, project.task_type)
    token, key = grant_import(body, project.id, user.id, settings)
    url, headers = store.presign_put(key, "application/zip", body.size, body.sha256)
    return UploadPermission(
        upload_token=token, upload_url=url, headers=headers, expires_in=UPLOAD_SECONDS
    )


@router.post("", response_model=ImportResponse)
def complete_import(
    body: CompleteRequest,
    project: OwnedProject,
    user: CurrentUser,
    settings: Configuration,
    session: Database,
    store: Storage,
    queue: Queue,
) -> ImportResponse:
    try:
        claim, key = read_import(
            body.upload_token.get_secret_value(), project.id, user.id, settings
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=400, detail="Invalid or expired import grant"
        ) from None
    check_task(claim, project.task_type)
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Another import is running; retry shortly",
            headers={"Retry-After": "5"},
        )
    try:
        with TemporaryFile() as source:
            try:
                store.download(key, source, claim.size, claim.sha256)
            except ClientError as error:
                if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
                    raise HTTPException(
                        status_code=409, detail="Upload the archive before importing it"
                    ) from None
                raise
            archive = Archive(source, claim.root)
            try:
                dataset = parse_dataset(archive, claim)
                return register_dataset(
                    dataset, archive, project, user, session, store, queue
                )
            finally:
                archive.close()
    except InvalidDataset as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except (
        ValueError,
        BadZipFile,
        LargeZipFile,
        yaml.YAMLError,
        RecursionError,
        zlib.error,
        EOFError,
        OverflowError,
        NotImplementedError,
    ):
        # Avoid reflecting decoder/library errors that can include internal paths.
        raise HTTPException(
            status_code=422,
            detail="Invalid archive, image or labels; check the import format and limits",
        ) from None
    finally:
        slots.release()
