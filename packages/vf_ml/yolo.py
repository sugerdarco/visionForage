"""Materialize a frozen detection manifest in Ultralytics YOLO format."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from vf_ml.manifest import parse_manifest_json


IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


def prepare_yolo_dataset(
    manifest_path: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    include_test: bool = False,
) -> dict:
    """Copy selected images and labels into a new, independently usable directory."""
    if type(include_test) is not bool:
        raise ValueError("include_test must be a boolean")
    contents = Path(manifest_path).read_bytes()
    manifest = parse_manifest_json(contents.decode("utf-8"))
    if manifest.task_type != "detection":
        raise ValueError("YOLO preparation requires a detection manifest")
    splits = ("train", "val", "test") if include_test else ("train", "val")
    for split in splits:
        if not manifest.samples_for(split):
            raise ValueError(f"Manifest has no samples for {split}")
    training_classes = {
        label.class_index
        for sample in manifest.samples_for("train")
        for label in sample.labels
    }
    if training_classes != set(range(len(manifest.class_names))):
        raise ValueError("Every manifest class must have a training box")
    data_root = Path(data_root).resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError("data_root must be a directory")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"YOLO dataset already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively, then replace our empty directory with the completed tree.
    output.mkdir()
    completed = False
    try:
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".yolo-") as work:
            staging = Path(work) / "dataset"
            staging.mkdir()
            for split in splits:
                (staging / "images" / split).mkdir(parents=True)
                (staging / "labels" / split).mkdir(parents=True)
            records = []
            for sample in manifest.samples:
                if sample.split not in splits:
                    continue
                source = (data_root / sample.image_key).resolve(strict=True)
                if not source.is_relative_to(data_root) or not source.is_file():
                    raise ValueError(
                        f"Image is not a file inside data_root: {sample.image_key}"
                    )
                suffix = Path(sample.image_key).suffix.lower()
                if suffix not in IMAGE_FORMATS:
                    raise ValueError(f"Expected JPEG or PNG image: {sample.image_key}")
                # Full object keys, not basenames, determine the destination names.
                identifier = hashlib.sha256(
                    sample.image_key.encode("utf-8")
                ).hexdigest()
                image_key = f"images/{sample.split}/{identifier}{suffix}"
                label_key = f"labels/{sample.split}/{identifier}.txt"
                target = staging / image_key
                shutil.copyfile(source, target)
                # Decode the actual copied bytes that training will consume.
                with Image.open(target) as image:
                    if image.format != IMAGE_FORMATS[suffix] or image.mode not in (
                        "RGB",
                        "L",
                    ):
                        raise ValueError(
                            f"Unsupported image encoding or mode: {sample.image_key}"
                        )
                    if min(image.size) < 10:
                        raise ValueError(
                            f"YOLO images must be at least 10x10: {sample.image_key}"
                        )
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError(
                            f"Normalize EXIF orientation and boxes before preparation: {sample.image_key}"
                        )
                    image.load()
                    width, height = image.size
                rows = []
                for label in sample.labels:
                    x, y, box_width, box_height = label.bbox
                    values = (
                        x + box_width / 2,
                        y + box_height / 2,
                        box_width,
                        box_height,
                    )
                    rows.append(
                        f"{label.class_index} "
                        + " ".join(format(value, ".17g") for value in values)
                    )
                # Explicit empty files distinguish known background images from missing labels.
                label_bytes = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
                (staging / label_key).write_bytes(label_bytes)
                with target.open("rb") as stream:
                    image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                records.append(
                    {
                        "image_key": sample.image_key,
                        "split": sample.split,
                        "image_file": image_key,
                        "label_file": label_key,
                        "image_sha256": image_hash,
                        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                        "width": width,
                        "height": height,
                        "boxes": len(sample.labels),
                    }
                )
            # JSON-quoted scalars and arrays are valid YAML, including unusual class names.
            # Absolute path avoids dependence on Ultralytics' global datasets directory.
            config = [f"path: {json.dumps(str(output.resolve()), ensure_ascii=True)}"]
            config.extend(f"{split}: images/{split}" for split in splits)
            config.append(
                f"names: {json.dumps(list(manifest.class_names), ensure_ascii=True)}"
            )
            (staging / "dataset.yaml").write_text(
                "\n".join(config) + "\n", encoding="utf-8"
            )
            (staging / "manifest.json").write_bytes(contents)
            report = {
                "schema_version": 1,
                "manifest_sha256": hashlib.sha256(contents).hexdigest(),
                "class_map": dict(manifest.class_map),
                "included_splits": list(splits),
                "image_policy": "Copy original bytes; no resizing, augmentation or EXIF rotation",
                "box_conversion": "Normalized top-left xywh to center xywh: [x+w/2, y+h/2, w, h]",
                "counts": {
                    split: {
                        "images": sum(row["split"] == split for row in records),
                        "boxes": sum(
                            row["boxes"] for row in records if row["split"] == split
                        ),
                    }
                    for split in splits
                },
                "images": records,
            }
            (staging / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            staging.replace(output)
            completed = True
    finally:
        if not completed:
            output.rmdir()
    return report
