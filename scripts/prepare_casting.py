"""Freeze casting splits after auditing decoded images for exact duplicates."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
import zipfile

import PIL
from PIL import Image

from scripts.download_datasets import CATALOG, prepare_dataset, sha256
from scripts.snapshots import publish_snapshot
from vf_ml.manifest import SPLITS, parse_manifest


CLASSES = {"ok_front": ("ok", 0), "def_front": ("defective", 1)}


def build_snapshot(
    root: Path, data_dir: Path, spec: dict, seed: int, val_fraction: float
) -> tuple[dict, dict]:
    """Audit a verified source tree, retaining test images over training copies."""
    if type(seed) is not int or not 0 < val_fraction < 1:
        raise ValueError(
            "Use an integer seed and a validation fraction between 0 and 1"
        )
    records = []
    groups = defaultdict(list)
    source = root / "casting_data" / "casting_data"
    for split in ("train", "test"):
        for folder, (name, index) in CLASSES.items():
            paths = sorted((source / split / folder).glob("*.jpeg"))
            if not paths:
                raise ValueError(f"No images in {split}/{folder}")
            for path in paths:
                with Image.open(path) as image:
                    if image.format != "JPEG" or image.size != (300, 300):
                        raise ValueError(f"Expected a 300x300 JPEG: {path}")
                    # Include dimensions and mode so this digest has explicit semantics.
                    pixels = hashlib.sha256(b"RGB:300:300\0")
                    pixels.update(image.convert("RGB").tobytes())
                record = {
                    "image_key": path.relative_to(data_dir).as_posix(),
                    "source_split": split,
                    "class_name": name,
                    "class_index": index,
                    "sha256": sha256(path),
                    "pixel_sha256": pixels.hexdigest(),
                    "split": split,
                }
                records.append(record)
                groups[record["pixel_sha256"]].append(record)

    duplicates = []
    for digest, members in sorted(groups.items()):
        if len({row["class_index"] for row in members}) != 1:
            raise ValueError(f"Identical pixels have conflicting labels: {digest}")
        if len(members) < 2:
            continue
        test = [row for row in members if row["source_split"] == "test"]
        keep = min(test or members, key=lambda row: row["image_key"])
        duplicates.append(
            {
                "pixel_sha256": digest,
                "image_keys": sorted(row["image_key"] for row in members),
            }
        )
        for row in members:
            if row["source_split"] == "train" and row is not keep:
                row["split"] = None
                row["excluded_reason"] = (
                    "test_overlap" if test else "training_duplicate"
                )
                row["duplicate_of"] = keep["image_key"]

    for name, index in CLASSES.values():
        candidates = [
            row
            for row in records
            if row["split"] == "train" and row["class_index"] == index
        ]
        if len(candidates) < 2:
            raise ValueError(f"Need at least two unique training images for {name}")
        # Hash ranking avoids dependence on filesystem order or random-library changes.
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['pixel_sha256']}".encode("ascii")
            ).digest()
        )
        count = max(
            1, min(len(candidates) - 1, int(len(candidates) * val_fraction + 0.5))
        )
        for row in candidates[:count]:
            row["split"] = "val"

    records.sort(key=lambda row: row["image_key"])
    manifest = {
        "schema_version": 1,
        "task_type": "classification",
        "class_map": {name: index for name, index in CLASSES.values()},
        "samples": [
            {
                "image_key": row["image_key"],
                "split": row["split"],
                "labels": [{"class_index": row["class_index"], "bbox": None}],
            }
            for row in records
            if row["split"] is not None
        ],
    }
    parse_manifest(manifest)
    report = {
        "preparation_version": 1,
        "source": spec,
        "pillow_version": PIL.__version__,
        "seed": seed,
        "validation_fraction_of_remaining_train": val_fraction,
        "split_method": "sha256(seed:pixel_sha256) ranking per class; nearest count, at least one per split",
        "duplicate_policy": "Preserve supplied test; exclude training copies of test, then keep one per training pixel hash",
        "limitations": "Exact RGB pixels only; augmented relatives may remain across splits because parent-image IDs are unavailable",
        "counts": {
            split: {
                name: sum(
                    row["split"] == split and row["class_name"] == name
                    for row in records
                )
                for name, _ in CLASSES.values()
            }
            for split in SPLITS
        },
        "excluded_images": sum(row["split"] is None for row in records),
        "duplicate_groups": duplicates,
        "images": records,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        help="New snapshot directory; default: DATA_DIR/manifests/casting-v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    try:
        spec = json.loads((CATALOG / "casting.json").read_text(encoding="utf-8"))
        root = prepare_dataset(spec, args.data_dir)
        manifest, report = build_snapshot(
            root, args.data_dir, spec, args.seed, args.val_fraction
        )
        output = args.output or args.data_dir / "manifests" / "casting-v2"
        publish_snapshot(output, manifest, report)
    except (OSError, ValueError, URLError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Casting preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": report["counts"],
                "excluded_images": report["excluded_images"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
