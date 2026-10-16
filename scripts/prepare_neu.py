"""Freeze NEU-DET detection splits while keeping identical pixels together."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
import xml.etree.ElementTree as ET
import zipfile

import PIL
from PIL import Image

from scripts.download_datasets import CATALOG, prepare_dataset, sha256
from scripts.snapshots import publish_snapshot
from vf_ml.manifest import SPLITS, parse_manifest


CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
CLASS_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def _text(node: ET.Element, field: str) -> str:
    elements = node.findall(field)
    if len(elements) != 1 or not elements[0].text or not elements[0].text.strip():
        raise ValueError(f"Expected exactly one nonempty XML field: {field}")
    return elements[0].text.strip()


def read_annotation(path: Path, image_path: Path) -> tuple[list, list, int]:
    """Convert inclusive one-based VOC boxes into normalized top-left xywh."""
    root = ET.parse(path).getroot()
    if root.tag != "annotation":
        raise ValueError(f"Expected an annotation root: {path}")
    if _text(root, "filename") not in (image_path.name, image_path.stem):
        raise ValueError(f"XML filename does not match image stem: {path}")
    width, height, depth = (
        int(_text(root, f"size/{key}")) for key in ("width", "height", "depth")
    )
    if (width, height) != (200, 200) or depth not in (1, 3):
        raise ValueError(f"Expected 200x200 XML dimensions and depth 1 or 3: {path}")
    labels, objects = [], []
    for obj in root.findall("object"):
        name = _text(obj, "name")
        if name not in CLASS_MAP:
            raise ValueError(f"Unknown object class {name}: {path}")
        difficult, truncated = (
            int(_text(obj, key)) for key in ("difficult", "truncated")
        )
        if difficult not in (0, 1) or truncated not in (0, 1):
            raise ValueError(f"Invalid difficult/truncated flag: {path}")
        xmin, ymin, xmax, ymax = (
            int(_text(obj, f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")
        )
        if not (1 <= xmin <= xmax <= width and 1 <= ymin <= ymax <= height):
            raise ValueError(f"Invalid one-based VOC bounding box: {path}")
        labels.append(
            {
                "class_index": CLASS_MAP[name],
                "bbox": [
                    (xmin - 1) / width,
                    (ymin - 1) / height,
                    (xmax - xmin + 1) / width,
                    (ymax - ymin + 1) / height,
                ],
            }
        )
        objects.append(
            {
                "class_name": name,
                "voc_xyxy": [xmin, ymin, xmax, ymax],
                "difficult": difficult,
                "truncated": truncated,
            }
        )
    if not labels:
        raise ValueError(f"NEU-DET annotation has no objects: {path}")
    return labels, objects, depth


def build_snapshot(
    root: Path,
    data_dir: Path,
    spec: dict,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[dict, dict]:
    """Audit a verified source tree and split pixel groups within filename families."""
    if (
        type(seed) is not int
        or not 0 < val_fraction < 1
        or not 0 < test_fraction < 1
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Use an integer seed and positive fractions whose sum is below 1"
        )
    source = root / "NEU-DET"
    images = {path.stem: path for path in (source / "IMAGES").glob("*.jpg")}
    annotations = {path.stem: path for path in (source / "ANNOTATIONS").glob("*.xml")}
    if not images or images.keys() != annotations.keys():
        raise ValueError("Image and XML stems must match one-to-one and be nonempty")
    records, groups = [], defaultdict(list)
    for stem, path in sorted(images.items()):
        family, separator, number = stem.rpartition("_")
        if not separator or family not in CLASS_MAP or not number.isdecimal():
            raise ValueError(f"Unrecognized NEU-DET filename family: {path}")
        with Image.open(path) as image:
            if (
                image.format != "JPEG"
                or image.size != (200, 200)
                or image.mode != "RGB"
            ):
                raise ValueError(f"Expected a 200x200 RGB JPEG: {path}")
            pixels = hashlib.sha256(b"RGB:200:200\0" + image.tobytes()).hexdigest()
        labels, objects, depth = read_annotation(annotations[stem], path)
        record = {
            "image_key": path.relative_to(data_dir).as_posix(),
            "annotation_key": annotations[stem].relative_to(data_dir).as_posix(),
            "sha256": sha256(path),
            "annotation_sha256": sha256(annotations[stem]),
            "pixel_sha256": pixels,
            "source_family": family,
            "xml_depth": depth,
            "objects": objects,
            "labels": labels,
        }
        records.append(record)
        groups[pixels].append(record)
    families = defaultdict(list)
    duplicates = []
    for digest, members in sorted(groups.items()):
        names = {row["source_family"] for row in members}
        if len(names) != 1:
            raise ValueError(
                f"Identical pixels span different filename families: {digest}"
            )
        families[members[0]["source_family"]].append((digest, members))
        if len(members) > 1:
            signatures = {
                json.dumps(
                    sorted(
                        row["objects"], key=lambda obj: json.dumps(obj, sort_keys=True)
                    ),
                    sort_keys=True,
                )
                for row in members
            }
            duplicates.append(
                {
                    "pixel_sha256": digest,
                    "image_keys": [row["image_key"] for row in members],
                    "annotations_differ": len(signatures) > 1,
                }
            )
    for name in CLASS_NAMES:
        candidates = families[name]
        count = len(candidates)
        if count < 3:
            raise ValueError(f"Need at least three unique pixel groups for {name}")
        val_count = max(1, int(count * val_fraction + 0.5))
        test_count = max(1, int(count * test_fraction + 0.5))
        if val_count + test_count >= count:
            raise ValueError(f"Fractions leave no training groups for {name}")
        candidates.sort(
            key=lambda pair: hashlib.sha256(
                f"{seed}:{pair[0]}".encode("ascii")
            ).digest()
        )
        for index, (_, members) in enumerate(candidates):
            split = (
                "test"
                if index < test_count
                else "val"
                if index < test_count + val_count
                else "train"
            )
            for row in members:
                row["split"] = split
    manifest = {
        "schema_version": 1,
        "task_type": "detection",
        "class_map": CLASS_MAP.copy(),
        "samples": [
            {
                "image_key": row["image_key"],
                "split": row["split"],
                "labels": row["labels"],
            }
            for row in records
        ],
    }
    parse_manifest(manifest)
    counts = {}
    for split in SPLITS:
        selected = [row for row in records if row["split"] == split]
        box_counts = {
            name: sum(
                obj["class_name"] == name for row in selected for obj in row["objects"]
            )
            for name in CLASS_NAMES
        }
        if not all(box_counts.values()):
            raise ValueError(f"Every object class must be represented in {split}")
        counts[split] = {
            "images": len(selected),
            "pixel_groups": len({row["pixel_sha256"] for row in selected}),
            "images_by_source_family": {
                name: sum(row["source_family"] == name for row in selected)
                for name in CLASS_NAMES
            },
            "images_by_object_class": {
                name: sum(
                    any(obj["class_name"] == name for obj in row["objects"])
                    for row in selected
                )
                for name in CLASS_NAMES
            },
            "boxes_by_class": box_counts,
            "difficult_boxes": sum(
                obj["difficult"] for row in selected for obj in row["objects"]
            ),
        }
    report = {
        "preparation_version": 1,
        "source": spec,
        "pillow_version": PIL.__version__,
        "seed": seed,
        "validation_fraction_of_pixel_groups": val_fraction,
        "test_fraction_of_pixel_groups": test_fraction,
        "split_method": "sha256(seed:pixel_sha256) ranking within filename family; rounded test groups first, then validation; rest train",
        "duplicate_policy": "Keep all images and original annotations; identical decoded RGB pixels stay in one split",
        "box_conversion": "VOC one-based inclusive xyxy to zero-based normalized top-left xywh: [(xmin-1)/W, (ymin-1)/H, (xmax-xmin+1)/W, (ymax-ymin+1)/H]",
        "difficult_policy": "Include difficult and truncated objects as ordinary targets in every split; flags retained here, not in manifest schema 1",
        "limitations": "Custom split, not an official benchmark; exact pixels only, no acquisition/parent IDs; stratified by filename family, not all object labels; difficult objects are not ignored for evaluation",
        "counts": counts,
        "duplicate_groups": duplicates,
        "images": records,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, help="Default: DATA_DIR/manifests/neu-det-v1"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    try:
        spec = json.loads((CATALOG / "neu-det.json").read_text(encoding="utf-8"))
        root = prepare_dataset(spec, args.data_dir)
        manifest, report = build_snapshot(
            root, args.data_dir, spec, args.seed, args.val_fraction, args.test_fraction
        )
        output = args.output or args.data_dir / "manifests" / "neu-det-v1"
        publish_snapshot(output, manifest, report)
    except (OSError, ValueError, ET.ParseError, URLError, zipfile.BadZipFile) as error:
        parser.exit(1, f"NEU-DET preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": report["counts"],
                "duplicate_groups": len(report["duplicate_groups"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
