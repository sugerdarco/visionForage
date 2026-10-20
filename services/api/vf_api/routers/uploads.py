"""Direct S3 uploads, verified registration and private image reads."""

import hashlib
import logging
from typing import Annotated, Literal
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, Query as QueryParameter, Response
import jwt
from kombu.exceptions import OperationalError as QueueError
from sqlalchemy import func, select

from vf_api.dependencies import (
    Configuration,
    CurrentUser,
    Database,
    OwnedProject,
    owned_project,
)
from vf_api.image_files import decode_image
from vf_api.models import Annotation, Image, Project
from vf_api.queueing import THUMBNAIL_TASK
from vf_api.storage import ObjectStore, READ_SECONDS, UPLOAD_SECONDS
from vf_api.upload_dependencies import Queue, Storage
from vf_api.upload_schemas import (
    CompleteRequest,
    ImagePage,
    ImageResponse,
    PresignRequest,
    UploadPermission,
)
from vf_api.upload_tokens import grant_upload, read_upload

router = APIRouter(tags=["images"])
log = logging.getLogger(__name__)


def image_response(image: Image, store: ObjectStore) -> ImageResponse:
    return ImageResponse(
        id=image.id,
        project_id=image.project_id,
        width=image.width,
        height=image.height,
        sha256=image.sha256,
        status=image.status,
        created_at=image.created_at,
        image_url=store.url(image.storage_key),
        thumbnail_url=store.url(image.thumb_key) if image.thumb_key else None,
        urls_expire_in=READ_SECONDS,
    )


def request_thumbnail(image: Image, queue) -> None:
    if image.thumb_key is None:
        try:
            queue.send_task(THUMBNAIL_TASK, args=[str(image.id)], retry=False)
        except (QueueError, OSError):
            # The periodic worker scan retries rows with no thumbnail after an outage.
            log.warning("Thumbnail enqueue unavailable; pending rows will be recovered")


@router.post(
    "/projects/{project_id}/uploads/presign", response_model=list[UploadPermission]
)
def presign(
    body: PresignRequest,
    project: OwnedProject,
    user: CurrentUser,
    settings: Configuration,
    store: Storage,
) -> list[UploadPermission]:
    permissions = []
    for file in body.files:
        token, key = grant_upload(file, project.id, user.id, settings)
        url, headers = store.presign_put(key, file.content_type, file.size, file.sha256)
        permissions.append(
            UploadPermission(
                upload_token=token,
                upload_url=url,
                headers=headers,
                expires_in=UPLOAD_SECONDS,
            )
        )
    return permissions


@router.post(
    "/projects/{project_id}/uploads/complete",
    response_model=ImageResponse,
    status_code=201,
)
def complete(
    body: CompleteRequest,
    response: Response,
    project: OwnedProject,
    user: CurrentUser,
    session: Database,
    settings: Configuration,
    store: Storage,
    queue: Queue,
) -> ImageResponse:
    try:
        file, staging_key = read_upload(
            body.upload_token.get_secret_value(),
            project.id,
            user.id,
            settings,
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=400, detail="Invalid or expired upload grant"
        ) from None

    statement = select(Image).where(
        Image.project_id == project.id, Image.sha256 == file.sha256
    )
    image = session.scalar(statement)
    if image is None:
        try:
            data = store.read(staging_key)
            if (
                len(data) != file.size
                or hashlib.sha256(data).hexdigest() != file.sha256
            ):
                raise ValueError("Uploaded size or SHA-256 does not match the grant")
            with decode_image(data, file.content_type) as decoded:
                width, height = decoded.size
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except ClientError as error:
            if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
                # A competing completion can commit and remove staging after our
                # initial lookup. Re-read before reporting an absent upload.
                image = session.scalar(statement)
                if image is not None:
                    response.status_code = 200
                    request_thumbnail(image, queue)
                    return image_response(image, store)
                raise HTTPException(
                    status_code=409, detail="Upload the file before completing it"
                ) from None
            raise

        # Serialize registration with deletion and other completions after decoding.
        owned_project(project.id, session, user, lock=True)
        image = session.scalar(statement)
        if image is None:
            suffix = "jpg" if file.content_type == "image/jpeg" else "png"
            key = f"{project.id}/originals/{file.sha256}.{suffix}"
            store.put(key, data, file.content_type)
            image = Image(
                project_id=project.id,
                storage_key=key,
                width=width,
                height=height,
                sha256=file.sha256,
            )
            session.add(image)
            session.commit()
        else:
            session.rollback()
            response.status_code = 200
    else:
        response.status_code = 200

    try:
        store.delete(staging_key)
    except (ClientError, BotoCoreError):
        log.warning("Temporary upload cleanup deferred to bucket lifecycle")
    request_thumbnail(image, queue)
    return image_response(image, store)


@router.get("/images/{image_id}", response_model=ImageResponse)
def read_image(
    image_id: UUID, user: CurrentUser, session: Database, store: Storage
) -> ImageResponse:
    image = session.scalar(
        select(Image)
        .join(Project)
        .where(Image.id == image_id, Project.owner_id == user.id)
    )
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return image_response(image, store)


@router.get("/projects/{project_id}/images", response_model=ImagePage)
def list_images(
    project: OwnedProject,
    session: Database,
    store: Storage,
    page: Annotated[int, QueryParameter(ge=1, le=1_000_000)] = 1,
    page_size: Annotated[int, QueryParameter(ge=1, le=100)] = 20,
    status: Literal["unlabeled", "labeled", "in_review"] | None = None,
    class_id: Annotated[UUID | None, QueryParameter(alias="class")] = None,
) -> ImagePage:
    filters = [Image.project_id == project.id]
    if status:
        filters.append(Image.status == status)
    if class_id:
        filters.append(
            select(Annotation.id)
            .where(
                Annotation.image_id == Image.id,
                Annotation.class_id == class_id,
            )
            .exists()
        )
    total = session.scalar(select(func.count()).select_from(Image).where(*filters))
    images = session.scalars(
        select(Image)
        .where(*filters)
        .order_by(Image.created_at.desc(), Image.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return ImagePage(
        items=[image_response(image, store) for image in images],
        page=page,
        page_size=page_size,
        total=total,
    )
