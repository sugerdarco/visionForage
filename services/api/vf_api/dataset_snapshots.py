"""Copy live metadata, then build deterministic manifests without ML dependencies."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath

from PIL import __version__ as pillow_version
from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.annotation_schemas import AnnotationInput
from vf_api.image_files import decode_image
from vf_api.models import Annotation, Class, Image, Project
from vf_api.project_schemas import Name
from vf_api.storage import ObjectStore
from vf_api.version_schemas import VersionCreate

SPLITS = ("train", "val", "test")
MAX_IMAGES = 10_000
MAX_CLASSES = 1000
MAX_LABELS = 100_000
MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_PIXELS = 1_000_000_000
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
SPLIT_METHOD = "sha256-rank-rgb-groups-v1"


class SnapshotError(ValueError):
    """Dataset cannot form a valid, bounded training snapshot."""


@dataclass
class Snapshot:
    task: str
    class_map: dict[str, int]
    images: list[dict]
    excluded: dict[str, int]


def copy_snapshot(project: Project, session: Session) -> Snapshot:
    """Caller holds the project lock; image locks prevent concurrent label edits."""
    if project.task_type not in ("classification", "detection"):
        raise SnapshotError("Task does not support dataset versions")
    classes = list(
        session.scalars(
            select(Class)
            .where(Class.project_id == project.id)
            .order_by(Class.index)
            .limit(MAX_CLASSES + 1)
        )
    )
    if len(classes) > MAX_CLASSES:
        raise SnapshotError("Version builds support at most 1,000 project classes")
    if not classes or [row.index for row in classes] != list(range(len(classes))):
        raise SnapshotError("Define a nonempty, contiguous project class map first")
    for row in classes:
        if TypeAdapter(Name).validate_python(row.name) != row.name:
            raise SnapshotError("Project class names are not normalized")
    indices = {row.id: row.index for row in classes}
    images = list(
        session.scalars(
            select(Image)
            .where(Image.project_id == project.id)
            .order_by(Image.id)
            .limit(MAX_IMAGES + 1)
            .with_for_update(read=True)
        )
    )
    if len(images) > MAX_IMAGES:
        raise SnapshotError("Version builds support at most 10,000 project images")
    selected = {row.id: row for row in images if row.status == "labeled"}
    if not selected:
        raise SnapshotError("Label images before creating a dataset version")
    labels = defaultdict(list)
    rows = session.scalars(
        select(Annotation)
        .where(Annotation.image_id.in_(selected))
        .limit(MAX_LABELS + 1)
    )
    for position, annotation in enumerate(rows):
        if position >= MAX_LABELS:
            raise SnapshotError("Version builds support at most 100,000 labels")
        if (
            annotation.class_id not in indices
            or annotation.source != "human"
            or annotation.confidence is not None
        ):
            raise SnapshotError(
                "Labeled images require reviewed human labels from this project"
            )
        validated = AnnotationInput(class_id=annotation.class_id, bbox=annotation.bbox)
        if (project.task_type == "classification") != (validated.bbox is None):
            raise SnapshotError("Labels do not match the project's task")
        labels[annotation.image_id].append(
            {
                "class_index": indices[annotation.class_id],
                "bbox": list(validated.bbox) if validated.bbox is not None else None,
            }
        )
    copied = []
    for identifier, row in selected.items():
        path = PurePosixPath(row.storage_key)
        if (
            not row.storage_key.startswith(f"{project.id}/originals/")
            or not row.storage_key.isprintable()
            or path.as_posix() != row.storage_key
            or ".." in path.parts
            or "\\" in row.storage_key
            or ":" in row.storage_key
        ):
            raise SnapshotError(
                "Original image key is outside the project image namespace"
            )
        targets = sorted(
            labels[identifier],
            key=lambda label: (label["class_index"], tuple(label["bbox"] or ())),
        )
        if project.task_type == "classification" and len(targets) != 1:
            raise SnapshotError(
                "Every labeled classification image needs exactly one class"
            )
        copied.append(
            {
                "image_id": str(identifier),
                "image_key": row.storage_key,
                "sha256": row.sha256,
                "width": row.width,
                "height": row.height,
                "labels": targets,
            }
        )
    return Snapshot(
        task=project.task_type,
        class_map={row.name: row.index for row in classes},
        images=sorted(copied, key=lambda row: row["image_key"]),
        excluded={
            status: sum(row.status == status for row in images)
            for status in ("unlabeled", "in_review")
        },
    )


def split_sizes(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 3:
        raise SnapshotError(
            "At least three unique pixel groups are required in each classification class, or overall for detection"
        )
    # Rounded evaluation counts, reserving at least one group for every split.
    test = min(total - 2, max(1, int(total * ratios["test"] + 0.5)))
    val = min(total - test - 1, max(1, int(total * ratios["val"] + 0.5)))
    return {"train": total - test - val, "val": val, "test": test}


def json_bytes(document: dict) -> bytes:
    data = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise SnapshotError("Snapshot document exceeds 32 MiB")
    return data


def build_snapshot(
    snapshot: Snapshot, options: VersionCreate, store: ObjectStore
) -> tuple[bytes, bytes, dict]:
    groups = defaultdict(list)
    source_bytes = pixels = 0
    for row in snapshot.images:
        data = store.read(row["image_key"])
        source_bytes += len(data)
        pixels += row["width"] * row["height"]
        if source_bytes > MAX_SOURCE_BYTES or pixels > MAX_PIXELS:
            raise SnapshotError(
                "Version exceeds 1 GiB of originals or one billion decoded pixels"
            )
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SnapshotError(
                "Original image bytes no longer match their registered checksum"
            )
        with decode_image(data) as image:
            if image.size != (row["width"], row["height"]):
                raise SnapshotError(
                    "Original image dimensions no longer match the database"
                )
            with image.convert("RGB") as rgb:
                digest = hashlib.sha256(
                    f"RGB:{rgb.width}:{rgb.height}\0".encode("ascii") + rgb.tobytes()
                ).hexdigest()
        row["pixel_sha256"] = digest
        groups[digest].append(row)

    strata = defaultdict(list)
    for digest, members in sorted(groups.items()):
        if snapshot.task == "classification":
            indices = {row["labels"][0]["class_index"] for row in members}
            if len(indices) != 1:
                raise SnapshotError(
                    "Identical decoded pixels have conflicting classification labels"
                )
            stratum = next(iter(indices))
        else:
            stratum = 0
        strata[stratum].append((digest, members))
    ratios = options.split_ratios.model_dump()
    for candidates in strata.values():
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{options.seed}:{item[0]}".encode("ascii")
            ).digest()
        )
        sizes = split_sizes(len(candidates), ratios)
        for position, (_, members) in enumerate(candidates):
            split = (
                "test"
                if position < sizes["test"]
                else "val"
                if position < sizes["test"] + sizes["val"]
                else "train"
            )
            for row in members:
                row["split"] = split

    manifest = {
        "schema_version": 1,
        "task_type": snapshot.task,
        "class_map": snapshot.class_map,
        "samples": [
            {key: row[key] for key in ("image_key", "split", "labels")}
            for row in snapshot.images
        ],
    }
    counts = {
        "images": {},
        "labels": {},
        "pixel_groups": {},
        "classes": {},
        "excluded": snapshot.excluded,
        "seed": options.seed,
        "split_method": SPLIT_METHOD,
        "duplicate_groups": sum(len(rows) > 1 for rows in groups.values()),
        "class_map": snapshot.class_map,
        "warnings": [],
    }
    for split in SPLITS:
        selected = [row for row in snapshot.images if row["split"] == split]
        counts["images"][split] = len(selected)
        counts["labels"][split] = sum(len(row["labels"]) for row in selected)
        counts["pixel_groups"][split] = len({row["pixel_sha256"] for row in selected})
        label_counts = Counter(
            label["class_index"] for row in selected for label in row["labels"]
        )
        counts["classes"][split] = {
            name: label_counts[index] for name, index in snapshot.class_map.items()
        }
        missing = [
            name for name, count in counts["classes"][split].items() if count == 0
        ]
        if missing and split == "train":
            raise SnapshotError(
                "Every project class needs a training example; add labeled groups or choose another seed"
            )
        if missing:
            counts["warnings"].append({"split": split, "missing_classes": missing})
    manifest_data = json_bytes(manifest)
    counts["manifest_sha256"] = hashlib.sha256(manifest_data).hexdigest()
    audit = {
        "schema_version": 1,
        "task_type": snapshot.task,
        "class_map": snapshot.class_map,
        "split_ratios": ratios,
        "counts": counts,
        "pillow_version": pillow_version,
        "images": [
            {key: value for key, value in row.items() if key != "labels"}
            for row in snapshot.images
        ],
    }
    audit_data = json_bytes(audit)
    counts["audit_sha256"] = hashlib.sha256(audit_data).hexdigest()
    return manifest_data, audit_data, counts
