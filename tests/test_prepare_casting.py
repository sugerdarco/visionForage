"""Exercise splitting and leakage checks with small synthetic JPEG fixtures."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.prepare_casting import build_snapshot, publish_snapshot
from vf_ml.manifest import load_manifest, parse_manifest


class PrepareCastingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.data = Path(self.work.name)
        self.root = self.data / "raw" / "casting-v2"
        self.source = self.root / "casting_data" / "casting_data"
        for class_id, folder in enumerate(("ok_front", "def_front")):
            for split, count in (("train", 6), ("test", 2)):
                directory = self.source / split / folder
                directory.mkdir(parents=True)
                for index in range(count):
                    color = (class_id * 120, index * 30, 200 if split == "test" else 0)
                    Image.new("RGB", (300, 300), color).save(
                        directory / f"{index}.jpeg"
                    )

    def build(self, seed: int = 42, fraction: float = 0.15) -> tuple[dict, dict]:
        return build_snapshot(self.root, self.data, {"name": "fixture"}, seed, fraction)

    def test_repeatable_stratified_splits_and_unchanged_test(self) -> None:
        manifest, report = self.build()
        self.assertEqual(self.build(), (manifest, report))
        self.assertEqual(
            report["counts"],
            {
                "train": {"ok": 5, "defective": 5},
                "val": {"ok": 1, "defective": 1},
                "test": {"ok": 2, "defective": 2},
            },
        )
        parsed = parse_manifest(manifest)
        self.assertEqual(
            {sample.image_key for sample in parsed.samples_for("test")},
            {
                path.relative_to(self.data).as_posix()
                for path in (self.source / "test").rglob("*.jpeg")
            },
        )
        self.assertNotEqual(self.build(seed=99)[0], manifest)
        self.assertTrue(
            all(not sample.image_key.startswith("/") for sample in parsed.samples)
        )

    def test_pixel_duplicates_are_excluded_even_when_jpeg_bytes_differ(self) -> None:
        original = (self.source / "test" / "ok_front" / "0.jpeg").read_bytes()
        # A JPEG comment changes the file checksum but leaves decoded pixels intact.
        copy = self.source / "train" / "ok_front" / "test-copy.jpeg"
        copy.write_bytes(original[:2] + b"\xff\xfe\x00\x06note" + original[2:])
        training = self.source / "train" / "def_front" / "0.jpeg"
        training.with_name("train-copy.jpeg").write_bytes(training.read_bytes())
        manifest, report = self.build()
        excluded = [row for row in report["images"] if row["split"] is None]
        self.assertEqual(report["excluded_images"], 2)
        self.assertEqual(
            {row["excluded_reason"] for row in excluded},
            {"test_overlap", "training_duplicate"},
        )
        self.assertEqual(len(manifest["samples"]), 16)
        hashes = {
            split: {
                row["pixel_sha256"] for row in report["images"] if row["split"] == split
            }
            for split in ("train", "val", "test")
        }
        self.assertFalse(hashes["train"] & hashes["test"])
        self.assertFalse(hashes["val"] & hashes["test"])
        self.assertFalse(hashes["train"] & hashes["val"])
        pair = [
            row
            for row in report["images"]
            if row["pixel_sha256"] == excluded[0]["pixel_sha256"]
        ]
        self.assertEqual(len(pair), 2)
        test_copy = next(
            row for row in excluded if row["excluded_reason"] == "test_overlap"
        )
        retained = next(
            row
            for row in report["images"]
            if row["image_key"] == test_copy["duplicate_of"]
        )
        self.assertNotEqual(test_copy["sha256"], retained["sha256"])

    def test_conflicting_labels_fail_before_creating_a_split(self) -> None:
        source = self.source / "train" / "ok_front" / "0.jpeg"
        (self.source / "test" / "def_front" / "0.jpeg").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            self.build()

    def test_duplicate_test_images_are_preserved_and_reported(self) -> None:
        source = self.source / "test" / "ok_front" / "0.jpeg"
        source.with_name("copy.jpeg").write_bytes(source.read_bytes())
        manifest, report = self.build()
        self.assertEqual(len(parse_manifest(manifest).samples_for("test")), 5)
        self.assertEqual(len(report["duplicate_groups"]), 1)
        self.assertEqual(report["excluded_images"], 0)

    def test_invalid_settings_and_missing_classes_fail(self) -> None:
        for fraction in (0, 1, -0.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.build(fraction=fraction)
        for path in (self.source / "train" / "ok_front").iterdir():
            path.unlink()
        with self.assertRaisesRegex(ValueError, "No images"):
            self.build()

    def test_corrupt_or_wrong_size_images_fail(self) -> None:
        path = self.source / "train" / "ok_front" / "0.jpeg"
        path.write_bytes(b"broken")
        with self.assertRaises(OSError):
            self.build()
        Image.new("RGB", (512, 512)).save(path)
        with self.assertRaisesRegex(ValueError, "300x300"):
            self.build()

    def test_published_snapshot_is_verified_on_rerun(self) -> None:
        manifest, report = self.build()
        output = self.data / "snapshot"
        publish_snapshot(output, manifest, report)
        publish_snapshot(output, manifest, report)
        self.assertEqual(
            load_manifest(output / "manifest.json"), parse_manifest(manifest)
        )
        saved = json.loads((output / "report.json").read_text())
        self.assertEqual(
            saved["manifest_sha256"],
            hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        )
        with self.assertRaisesRegex(ValueError, "choose a new"):
            publish_snapshot(output, *self.build(seed=99))
        (output / "report.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "choose a new"):
            publish_snapshot(output, manifest, report)
        self.assertEqual((output / "report.json").read_text(), "changed")


if __name__ == "__main__":
    unittest.main()
