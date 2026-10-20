"""Deterministic pixel-group splits and the version request contract."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic import ValidationError

from vf_api.dataset_snapshots import (
    Snapshot,
    SnapshotError,
    build_snapshot,
    split_sizes,
)
from vf_api.version_schemas import SplitRatios, VersionCreate


def picture(number, comment=""):
    output = BytesIO()
    info = PngInfo()
    info.add_text("comment", comment)
    with Image.new("RGB", (20, 10), color=(number, number * 2, number * 3)) as image:
        image.save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def snapshot_fixture(task="classification", count=12):
    objects = {}
    rows = []
    for index in range(count):
        data = picture(index)
        key = f"project/originals/{index}.png"
        objects[key] = data
        labels = (
            [{"class_index": index % 2, "bbox": None}]
            if task == "classification"
            else [
                {"class_index": 0, "bbox": [0, 0, 1, 1]},
                {"class_index": 1, "bbox": [0, 0, 0.5, 0.5]},
            ]
        )
        rows.append(
            {
                "image_id": str(index),
                "image_key": key,
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": 20,
                "height": 10,
                "labels": labels,
            }
        )
    return Snapshot(
        task, {"ok": 0, "defect": 1}, rows, {"unlabeled": 2, "in_review": 1}
    ), objects


class SnapshotTest(unittest.TestCase):
    def build(self, snapshot, objects, **options):
        return build_snapshot(
            snapshot,
            VersionCreate(**options),
            SimpleNamespace(read=objects.__getitem__),
        )

    def test_determinism_seed_counts_and_hashes(self):
        snapshot, objects = snapshot_fixture()
        first = self.build(deepcopy(snapshot), objects)
        self.assertEqual(first, self.build(deepcopy(snapshot), objects))
        changed = self.build(deepcopy(snapshot), objects, seed=7)
        self.assertNotEqual(first[0], changed[0])
        manifest, audit, counts = first
        self.assertEqual(
            hashlib.sha256(manifest).hexdigest(), counts["manifest_sha256"]
        )
        self.assertEqual(hashlib.sha256(audit).hexdigest(), counts["audit_sha256"])
        self.assertEqual(counts["images"], {"train": 8, "val": 2, "test": 2})
        self.assertEqual(counts["classes"]["train"], {"ok": 4, "defect": 4})
        self.assertEqual(counts["excluded"], {"unlabeled": 2, "in_review": 1})
        self.assertEqual(
            set(json.loads(manifest)),
            {"schema_version", "task_type", "class_map", "samples"},
        )
        self.assertEqual(len(json.loads(audit)["images"]), 12)

    def test_identical_pixels_stay_together_despite_different_png_bytes(self):
        snapshot, objects = snapshot_fixture()
        copy = deepcopy(snapshot.images[0])
        data = picture(0, "different encoding metadata")
        copy.update(
            image_id="copy",
            image_key="project/originals/copy.png",
            sha256=hashlib.sha256(data).hexdigest(),
        )
        objects[copy["image_key"]] = data
        snapshot.images.append(copy)
        _, audit, counts = self.build(snapshot, objects)
        rows = json.loads(audit)["images"]
        self.assertNotEqual(rows[0]["sha256"], rows[-1]["sha256"])
        self.assertEqual(rows[0]["pixel_sha256"], rows[-1]["pixel_sha256"])
        self.assertEqual(rows[0]["split"], rows[-1]["split"])
        self.assertEqual(counts["duplicate_groups"], 1)
        self.assertEqual(sum(counts["pixel_groups"].values()), 12)
        self.assertEqual(sum(counts["images"].values()), 13)
        snapshot.images[-1]["labels"][0]["class_index"] = 1
        with self.assertRaisesRegex(SnapshotError, "conflicting classification"):
            self.build(snapshot, objects)

    def test_detection_backgrounds_and_duplicate_pixel_annotations_are_retained(self):
        snapshot, objects = snapshot_fixture("detection")
        snapshot.images[0]["labels"] = []
        copy = deepcopy(snapshot.images[0])
        data = picture(0, "same pixels, separate annotations")
        copy.update(
            image_id="copy",
            image_key="project/originals/copy.png",
            sha256=hashlib.sha256(data).hexdigest(),
            labels=[{"class_index": 1, "bbox": [0, 0, 1, 1]}],
        )
        snapshot.images.append(copy)
        objects[copy["image_key"]] = data
        manifest, _, counts = self.build(snapshot, objects)
        rows = json.loads(manifest)["samples"]
        self.assertEqual(rows[0]["labels"], [])
        self.assertEqual(rows[-1]["labels"], copy["labels"])
        self.assertEqual(rows[0]["split"], rows[-1]["split"])
        self.assertEqual(counts["warnings"], [])

    def test_small_datasets_unused_classes_and_missing_evaluation_classes(self):
        snapshot, objects = snapshot_fixture(count=4)
        with self.assertRaisesRegex(SnapshotError, "three unique pixel groups"):
            self.build(snapshot, objects)
        snapshot, objects = snapshot_fixture()
        snapshot.class_map["unused"] = 2
        with self.assertRaisesRegex(SnapshotError, "Every project class"):
            self.build(snapshot, objects)
        snapshot, objects = snapshot_fixture("detection")
        manifest, _, _ = self.build(deepcopy(snapshot), objects)
        train_key = next(
            row["image_key"]
            for row in json.loads(manifest)["samples"]
            if row["split"] == "train"
        )
        for row in snapshot.images:
            if row["image_key"] != train_key:
                row["labels"] = row["labels"][:1]
        _, _, counts = self.build(snapshot, objects)
        self.assertEqual(
            counts["warnings"],
            [
                {"split": "val", "missing_classes": ["defect"]},
                {"split": "test", "missing_classes": ["defect"]},
            ],
        )

    def test_bad_originals_dimensions_and_resource_limits(self):
        snapshot, objects = snapshot_fixture()
        original = snapshot.images[0]["sha256"]
        snapshot.images[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(SnapshotError, "checksum"):
            self.build(snapshot, objects)
        snapshot.images[0]["sha256"] = original
        snapshot.images[0]["width"] = 21
        with self.assertRaisesRegex(SnapshotError, "dimensions"):
            self.build(snapshot, objects)
        snapshot.images[0]["width"] = 20
        for name in ("MAX_SOURCE_BYTES", "MAX_PIXELS", "MAX_DOCUMENT_BYTES"):
            with patch("vf_api.dataset_snapshots." + name, 1):
                with self.assertRaises(SnapshotError):
                    self.build(snapshot, objects)

    def test_split_rounding_and_strict_options(self):
        self.assertEqual(
            split_sizes(3, SplitRatios().model_dump()),
            {"train": 1, "val": 1, "test": 1},
        )
        self.assertEqual(
            split_sizes(20, {"train": 0.5, "val": 0.25, "test": 0.25}),
            {"train": 10, "val": 5, "test": 5},
        )
        for body in (
            {"seed": True},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": "42"},
            {"extra": 1},
            {"split_ratios": {"train": 0.8, "val": 0.15, "test": 0.15}},
            {"split_ratios": {"train": 1, "val": 0, "test": 0}},
            {"split_ratios": {"train": "0.7", "val": 0.15, "test": 0.15}},
            {"split_ratios": {"train": float("nan"), "val": 0.15, "test": 0.15}},
        ):
            with self.assertRaises(ValidationError):
                VersionCreate(**body)
