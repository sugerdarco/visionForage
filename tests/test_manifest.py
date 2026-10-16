"""Validate the training contract without image files, a database, or a GPU."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

from vf_ml.manifest import load_manifest, parse_manifest


def example(task: str = "classification") -> dict:
    return {
        "schema_version": 1,
        "task_type": task,
        "class_map": {"defective": 1, "ok": 0},
        "samples": [
            {
                "image_key": "project/image.jpg",
                "split": "train",
                "labels": [
                    {
                        "class_index": 1,
                        "bbox": None if task == "classification" else [0, 0, 1, 1],
                    }
                ],
            }
        ],
    }


class ManifestTest(unittest.TestCase):
    def test_class_indices_do_not_depend_on_json_field_order(self) -> None:
        manifest = parse_manifest(example())
        self.assertEqual(manifest.class_names, ("ok", "defective"))
        self.assertEqual(manifest.class_map["defective"], 1)
        self.assertEqual(manifest.samples_for("train")[0].labels[0].class_index, 1)
        self.assertEqual(manifest.samples_for("val"), ())
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            manifest.samples_for("validation")

    def test_detection_supports_multiple_classes_and_negative_images(self) -> None:
        data = example("detection")
        data["samples"][0]["labels"].append(
            {"class_index": 0, "bbox": [0.2, 0.3, 0.4, 0.5]}
        )
        data["samples"].append(
            {"image_key": "project/negative.jpg", "split": "test", "labels": []}
        )
        manifest = parse_manifest(data)
        self.assertEqual(len(manifest.samples_for("train")[0].labels), 2)
        self.assertEqual(manifest.samples_for("test")[0].labels, ())

    def test_loaded_records_are_immutable_copies_and_can_be_pickled(self) -> None:
        data = example("detection")
        manifest = parse_manifest(data)
        data["class_map"]["ok"] = 8
        data["samples"][0]["labels"][0]["bbox"][0] = 0.5
        self.assertEqual(manifest.class_map["ok"], 0)
        self.assertEqual(manifest.samples[0].labels[0].bbox, (0, 0, 1, 1))
        with self.assertRaises(FrozenInstanceError):
            manifest.samples[0].split = "test"
        with self.assertRaises(TypeError):
            manifest.class_map["ok"] = 8
        self.assertEqual(pickle.loads(pickle.dumps(manifest)), manifest)

    def test_schema_errors_are_rejected(self) -> None:
        for field, value in (
            ("schema_version", 2),
            ("schema_version", True),
            ("task_type", "anomaly"),
            ("samples", []),
        ):
            with self.subTest(field=field, value=value):
                data = example()
                data[field] = value
                with self.assertRaises(ValueError):
                    parse_manifest(data)
        for value in (None, [], {}, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_manifest(value)

    def test_unknown_and_missing_fields_are_rejected(self) -> None:
        for location in ("manifest", "sample", "label"):
            for mutation in ("add", "remove"):
                with self.subTest(location=location, mutation=mutation):
                    data = example()
                    target = data if location == "manifest" else data["samples"][0]
                    if location == "label":
                        target = target["labels"][0]
                    if mutation == "add":
                        target["typo"] = 1
                    else:
                        target.pop(next(iter(target)))
                    with self.assertRaisesRegex(ValueError, "expected fields"):
                        parse_manifest(data)

    def test_invalid_class_maps_are_rejected(self) -> None:
        for classes in (
            {},
            [],
            {"ok": True},
            {"ok": 1},
            {"ok": -1},
            {"ok": 0, "defective": 0},
            {" ok": 0},
            {"": 0},
        ):
            with self.subTest(classes=classes):
                data = example()
                data["class_map"] = classes
                with self.assertRaises(ValueError):
                    parse_manifest(data)

    def test_repeated_image_keys_are_rejected_within_and_across_splits(self) -> None:
        for split in ("train", "test"):
            with self.subTest(split=split):
                data = example()
                duplicate = deepcopy(data["samples"][0])
                duplicate["split"] = split
                data["samples"].append(duplicate)
                with self.assertRaisesRegex(ValueError, "duplicate image_key"):
                    parse_manifest(data)

    def test_unsafe_image_keys_and_unknown_splits_are_rejected(self) -> None:
        for key in (
            "",
            ".",
            "/image.jpg",
            "../image.jpg",
            "a/../image.jpg",
            "a//image.jpg",
            "a/./image.jpg",
            "a\\image.jpg",
            "s3://bucket/image.jpg",
            "a/",
            "a\x00.jpg",
        ):
            with self.subTest(key=key):
                data = example()
                data["samples"][0]["image_key"] = key
                with self.assertRaises(ValueError):
                    parse_manifest(data)
        data = example()
        data["samples"][0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split"):
            parse_manifest(data)

    def test_classification_requires_one_label_without_a_box(self) -> None:
        for labels in (
            [],
            None,
            [{"class_index": 0, "bbox": [0, 0, 1, 1]}],
            [{"class_index": 0, "bbox": None}] * 2,
        ):
            with self.subTest(labels=labels):
                data = example()
                data["samples"][0]["labels"] = labels
                with self.assertRaises(ValueError):
                    parse_manifest(data)

    def test_unknown_or_noninteger_class_indices_are_rejected(self) -> None:
        for index in (-1, 2, "1", True, 1.0):
            with self.subTest(index=index):
                data = example()
                data["samples"][0]["labels"][0]["class_index"] = index
                with self.assertRaisesRegex(ValueError, "class_index"):
                    parse_manifest(data)

    def test_detection_boxes_must_be_finite_normalized_and_inside_image(self) -> None:
        boxes = (
            None,
            [0, 0, 1],
            [True, 0, 1, 1],
            [float("nan"), 0, 1, 1],
            [0, 0, float("inf"), 1],
            [-0.1, 0, 1, 1],
            [0, 0, 0, 1],
            [0.9, 0, 0.2, 1],
            [0, 0.9, 1, 0.2],
        )
        for box in boxes:
            with self.subTest(box=box):
                data = example("detection")
                data["samples"][0]["labels"][0]["bbox"] = box
                with self.assertRaisesRegex(ValueError, "bbox"):
                    parse_manifest(data)

    def test_json_loader_rejects_duplicate_fields_and_nonstandard_numbers(self) -> None:
        valid = json.dumps(example())
        invalid = (
            valid.replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ),
            valid.replace('"ok": 0', '"ok": 0, "ok": 1'),
            valid.replace("null", "NaN"),
            "{broken json",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_manifest(path)

    def test_json_loader_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(example()), encoding="utf-8")
            self.assertEqual(load_manifest(path), parse_manifest(example()))
            command = [sys.executable, "-m", "vf_ml.manifest", str(path)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(
                json.loads(result.stdout)["images_by_split"],
                {"train": 1, "val": 0, "test": 0},
            )
            path.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Invalid manifest:", result.stderr)


if __name__ == "__main__":
    unittest.main()
