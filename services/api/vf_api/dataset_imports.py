"""Register a fully validated archive in one PostgreSQL transaction."""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.dependencies import owned_project
from vf_api.import_archive import Archive
from vf_api.import_formats import Dataset, Label
from vf_api.import_schemas import ImportedImage, ImportResponse
from vf_api.models import Annotation, Class, Image, Project, User
from vf_api.project_schemas import ClassCreate
from vf_api.routers.projects import append_classes
from vf_api.routers.uploads import request_thumbnail
from vf_api.storage import MAX_UPLOAD_BYTES, ObjectStore


def register_dataset(
    dataset: Dataset,
    archive: Archive,
    project: Project,
    user: User,
    session: Session,
    store: ObjectStore,
    queue,
) -> ImportResponse:
    # Same lock order as upload completion: project, then images. Label saves only
    # lock their own image; holding these locks protects the conflict preflight.
    owned_project(project.id, session, user, lock=True)
    unique = {sample.sha256: sample for sample in dataset.samples}
    existing = {
        image.sha256: image
        for image in session.scalars(
            select(Image)
            .where(Image.project_id == project.id, Image.sha256.in_(unique))
            .order_by(Image.id)
            .with_for_update()
        )
    }
    labels = {image.id: [] for image in existing.values()}
    machine_labels = set()
    for row, name in session.execute(
        select(Annotation, Class.name)
        .join(Class, Annotation.class_id == Class.id)
        .where(Annotation.image_id.in_(labels))
    ):
        labels[row.image_id].append(Label(name, tuple(row.bbox or ())))
        if row.source != "human" or row.confidence is not None:
            machine_labels.add(row.image_id)

    actions = {}
    for sha256, sample in unique.items():
        image = existing.get(sha256)
        if image is None:
            actions[sha256] = "created"
        elif image.status == "unlabeled" and not labels[image.id]:
            actions[sha256] = "labeled"
        elif (
            image.status == "labeled"
            and image.id not in machine_labels
            and sorted(labels[image.id]) == sample.labels
        ):
            actions[sha256] = "unchanged"
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Existing image has conflicting labels or review status: {sample.path}",
            )

    classes = {
        row.name: row
        for row in session.scalars(select(Class).where(Class.project_id == project.id))
    }
    added = append_classes(
        project,
        [
            ClassCreate(name=name, color="#3b82f6")
            for name in dataset.names
            if name not in classes
        ],
        session,
    )
    session.flush()
    classes.update({row.name: row for row in added})
    for sha256, sample in unique.items():
        action = actions[sha256]
        if action == "created":
            suffix = "jpg" if sample.mime == "image/jpeg" else "png"
            key = f"{project.id}/originals/{sha256}.{suffix}"
            store.put(key, archive.read(sample.path, MAX_UPLOAD_BYTES), sample.mime)
            image = Image(
                project_id=project.id,
                storage_key=key,
                width=sample.width,
                height=sample.height,
                sha256=sha256,
                status="labeled",
            )
            session.add(image)
            session.flush()
            existing[sha256] = image
        else:
            image = existing[sha256]
        if action != "unchanged":
            image.status = "labeled"
            session.add_all(
                [
                    Annotation(
                        image_id=image.id,
                        class_id=classes[label.name].id,
                        bbox=list(label.bbox) if label.bbox else None,
                        source="human",
                        confidence=None,
                    )
                    for label in sample.labels
                ]
            )

    seen = set()
    items = []
    for sample in dataset.samples:
        items.append(
            ImportedImage(
                path=sample.path,
                image_id=existing[sample.sha256].id,
                action="duplicate" if sample.sha256 in seen else actions[sample.sha256],
            )
        )
        seen.add(sample.sha256)
    response = ImportResponse(
        project_id=project.id,
        items=items,
        created_images=list(actions.values()).count("created"),
        labeled_images=list(actions.values()).count("labeled"),
        unchanged_images=list(actions.values()).count("unchanged"),
        duplicate_files=len(dataset.samples) - len(unique),
        created_classes=len(added),
    )
    session.commit()
    for image in existing.values():
        request_thumbnail(image, queue)
    return response
