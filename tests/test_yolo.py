"""Verify geometry and split isolation in materialized YOLO datasets."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import yaml

from vf_ml.yolo import prepare_yolo_dataset


class YoloDatasetTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.data = self.root / "source"
        self.data.mkdir()
        self.manifest_path = self.root / "manifest.json"
        self.output = self.root / 'yolo "quoted" path'
        self.manifest = {
            "schema_version": 1,
            "task_type": "detection",
            "class_map": {"a: # defect": 0, 'b "mark"': 1},
            "samples": [
                {
                    "image_key": "first/same.png",
                    "split": "train",
                    "labels": [
                        {"class_index": 0, "bbox": [0, 0, 1, 1]},
                        {"class_index": 1, "bbox": [0.1, 0.2, 0.3, 0.4]},
                    ],
                },
                {"image_key": "second/same.png", "split": "train", "labels": []},
                {
                    "image_key": "val.png",
                    "split": "val",
                    "labels": [
                        {"class_index": 1, "bbox": [0.995, 0.995, 0.005, 0.005]},
                    ],
                },
                {
                    "image_key": "missing-test.png",
                    "split": "test",
                    "labels": [
                        {"class_index": 0, "bbox": [0.5, 0.5, 0.5, 0.5]},
                    ],
                },
            ],
        }
        for index, row in enumerate(self.manifest["samples"][:3]):
            path = self.data / row["image_key"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (200, 100), (index * 70, 20, 30)).save(path)
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest) + "\n")

    def prepare(self, **kwargs):
        return prepare_yolo_dataset(
            self.manifest_path, self.data, self.output, **kwargs
        )

    def test_labels_names_provenance_and_default_test_isolation(self):
        report = self.prepare()
        config = yaml.safe_load((self.output / "dataset.yaml").read_text())
        self.assertEqual(config["path"], str(self.output))
        self.assertEqual(config["names"], ["a: # defect", 'b "mark"'])
        self.assertNotIn("test", config)
        self.assertFalse((self.output / "images/test").exists())
        self.assertEqual(
            report["counts"],
            {"train": {"images": 2, "boxes": 2}, "val": {"images": 1, "boxes": 1}},
        )
        self.assertEqual(
            (self.output / "manifest.json").read_bytes(),
            self.manifest_path.read_bytes(),
        )
        self.assertEqual(
            report["manifest_sha256"],
            hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        )
        rows = report["images"]
        self.assertEqual(len({row["image_file"] for row in rows}), 3)
        boxes = [
            [float(x) for x in line.split()]
            for line in (self.output / rows[0]["label_file"]).read_text().splitlines()
        ]
        self.assertEqual(boxes, [[0, 0.5, 0.5, 1, 1], [1, 0.25, 0.4, 0.3, 0.4]])
        edge = [
            float(x) for x in (self.output / rows[2]["label_file"]).read_text().split()
        ]
        self.assertAlmostEqual(edge[1], 0.9975)
        self.assertAlmostEqual(edge[2], 0.9975)
        self.assertEqual(edge[3:], [0.005, 0.005])
        self.assertEqual((self.output / rows[1]["label_file"]).read_bytes(), b"")
        for row in rows:
            image = self.output / row["image_file"]
            self.assertEqual(
                image.read_bytes(), (self.data / row["image_key"]).read_bytes()
            )
            self.assertFalse(image.is_symlink())
            self.assertEqual(
                row["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest()
            )
            self.assertEqual(
                row["label_sha256"],
                hashlib.sha256(
                    (self.output / row["label_file"]).read_bytes()
                ).hexdigest(),
            )
        self.assertEqual(json.loads((self.output / "report.json").read_text()), report)

    def test_explicit_test_materialization_and_missing_image_cleanup(self):
        with self.assertRaises(FileNotFoundError):
            self.prepare(include_test=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".yolo-*")), [])
        Image.new("L", (20, 30)).save(self.data / "missing-test.png")
        report = self.prepare(include_test=True)
        self.assertEqual(report["counts"]["test"], {"images": 1, "boxes": 1})
        self.assertEqual(
            yaml.safe_load((self.output / "dataset.yaml").read_text())["test"],
            "images/test",
        )

    def test_copies_are_independent_and_existing_outputs_are_refused(self):
        report = self.prepare()
        row = report["images"][0]
        target = self.output / row["image_file"]
        original = target.read_bytes()
        (self.data / row["image_key"]).write_bytes(b"changed source")
        self.assertEqual(target.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(target.read_bytes(), original)

    def test_existing_empty_directory_and_symlink_are_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.output.rmdir()
        self.output.symlink_to(self.root / "missing-output")
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_path_escape_and_symlink_escape_publish_nothing(self):
        self.manifest["samples"][0]["image_key"] = "../outside.png"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "relative object key"):
            self.prepare()
        self.manifest["samples"][0]["image_key"] = "escape.png"
        self.save_manifest()
        outside = self.root / "outside.png"
        Image.new("RGB", (20, 20)).save(outside)
        (self.data / "escape.png").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "inside data_root"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_images_and_orientation_publish_nothing(self):
        path = self.data / "first/same.png"
        path.write_bytes(b"broken")
        with self.assertRaises(OSError):
            self.prepare()
        self.assertFalse(self.output.exists())
        for image in (Image.new("RGB", (5, 20)), Image.new("RGBA", (20, 20))):
            image.save(path)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (20, 30)).save(path, exif=exif)
        with self.assertRaisesRegex(ValueError, "EXIF"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_task_missing_splits_classes_and_bad_boxes(self):
        original = deepcopy(self.manifest)
        self.manifest["samples"] = [
            row for row in self.manifest["samples"] if row["split"] != "val"
        ]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "no samples for val"):
            self.prepare()
        self.manifest = deepcopy(original)
        self.manifest["samples"][0]["labels"] = []
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "training box"):
            self.prepare()
        self.manifest = deepcopy(original)
        self.manifest["samples"][0]["labels"][0]["bbox"] = [0, 0, 2, 1]
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.prepare()
        self.manifest = deepcopy(original)
        self.manifest["task_type"] = "classification"
        for row in self.manifest["samples"]:
            row["labels"] = [{"class_index": 0, "bbox": None}]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "detection manifest"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_import_does_not_load_training_libraries(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import vf_ml.yolo; assert not {'torch', 'timm', 'ultralytics'} & sys.modules.keys()",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
