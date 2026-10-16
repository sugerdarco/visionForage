"""Check VOC geometry, split isolation, and annotation provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from scripts.prepare_neu import CLASS_MAP, build_snapshot, read_annotation
from scripts.snapshots import publish_snapshot
from vf_ml.manifest import load_manifest, parse_manifest


class PrepareNeuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.data = Path(self.work.name)
        self.root = self.data / "raw" / "neu-det-v1"
        self.images = self.root / "NEU-DET" / "IMAGES"
        self.annotations = self.root / "NEU-DET" / "ANNOTATIONS"
        self.images.mkdir(parents=True)
        self.annotations.mkdir()
        for name, class_id in CLASS_MAP.items():
            for index in range(10):
                stem = f"{name}_{index}"
                Image.new("RGB", (200, 200), (class_id * 35, index * 20, 40)).save(
                    self.images / f"{stem}.jpg"
                )
                self.write_xml(stem, name)

    def write_xml(self, stem, name, box=(1, 1, 200, 200)):
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = stem + ".jpg"
        size = ET.SubElement(root, "size")
        for key, value in (("width", 200), ("height", 200), ("depth", 1)):
            ET.SubElement(size, key).text = str(value)
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "truncated").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bbox, key).text = str(value)
        ET.ElementTree(root).write(self.annotations / f"{stem}.xml")

    def build(self, **kwargs):
        return build_snapshot(self.root, self.data, {"name": "fixture"}, **kwargs)

    def test_repeatable_splits_counts_and_relative_keys(self) -> None:
        manifest, report = self.build()
        self.assertEqual((manifest, report), self.build())
        self.assertNotEqual(manifest, self.build(seed=99)[0])
        parsed = parse_manifest(manifest)
        self.assertEqual(parsed.task_type, "detection")
        for split, count in (("train", 36), ("val", 12), ("test", 12)):
            self.assertEqual(len(parsed.samples_for(split)), count)
            self.assertEqual(
                report["counts"][split]["boxes_by_class"],
                {name: count // 6 for name in CLASS_MAP},
            )
        self.assertTrue(
            all(not sample.image_key.startswith("/") for sample in parsed.samples)
        )
        self.assertEqual(len({sample.image_key for sample in parsed.samples}), 60)

    def test_inclusive_voc_edges_single_pixel_and_actual_object_classes(self) -> None:
        path = self.annotations / "crazing_0.xml"
        root = ET.parse(path).getroot()
        root.find("filename").text = "crazing_0"  # Some upstream filenames omit .jpg.
        obj = deepcopy(root.find("object"))
        obj.find("name").text = "scratches"  # Filename prefix must not label this box.
        obj.find("difficult").text = "1"
        obj.find("truncated").text = "1"
        for key in ("xmin", "xmax", "ymin", "ymax"):
            obj.find(f"bndbox/{key}").text = "200"
        root.append(obj)
        ET.ElementTree(root).write(path)
        manifest, report = self.build()
        row = next(
            row
            for row in manifest["samples"]
            if row["image_key"].endswith("/crazing_0.jpg")
        )
        self.assertEqual(
            row["labels"],
            [
                {"class_index": 0, "bbox": [0, 0, 1, 1]},
                {"class_index": 5, "bbox": [0.995, 0.995, 0.005, 0.005]},
            ],
        )
        record = next(
            row
            for row in report["images"]
            if row["image_key"].endswith("/crazing_0.jpg")
        )
        self.assertEqual(record["objects"][1]["difficult"], 1)
        self.assertEqual(record["objects"][1]["truncated"], 1)
        self.assertEqual(record["xml_depth"], 1)
        self.assertEqual(
            record["annotation_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertEqual(
            sum(row["difficult_boxes"] for row in report["counts"].values()), 1
        )

    def test_pixel_duplicates_with_different_boxes_share_one_split(self) -> None:
        original = (self.images / "patches_0.jpg").read_bytes()
        # JPEG comment changes bytes without changing decoded pixels.
        (self.images / "patches_99.jpg").write_bytes(
            original[:2] + b"\xff\xfe\x00\x06note" + original[2:]
        )
        self.write_xml("patches_99", "patches", (2, 3, 190, 195))
        manifest, report = self.build()
        self.assertEqual(len(manifest["samples"]), 61)
        self.assertEqual(len(report["duplicate_groups"]), 1)
        group = report["duplicate_groups"][0]
        self.assertTrue(group["annotations_differ"])
        members = [
            row for row in report["images"] if row["image_key"] in group["image_keys"]
        ]
        self.assertEqual(len({row["split"] for row in members}), 1)
        self.assertEqual(len({row["sha256"] for row in members}), 2)
        splits = {
            split: {
                row["pixel_sha256"] for row in report["images"] if row["split"] == split
            }
            for split in ("train", "val", "test")
        }
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["val"] & splits["test"])
        self.assertFalse(splits["train"] & splits["val"])

    def test_invalid_boxes_and_xml_fields_are_rejected(self) -> None:
        path = self.annotations / "crazing_0.xml"
        original = path.read_bytes()
        for field, value in (
            ("object/bndbox/xmin", "0"),
            ("object/bndbox/xmax", "201"),
            ("object/bndbox/xmin", "201"),
            ("object/bndbox/ymax", "nan"),
            ("object/bndbox/ymin", "1.5"),
            ("object/name", "unknown"),
            ("object/difficult", "2"),
            ("size/width", "201"),
            ("filename", "../other.jpg"),
        ):
            root = ET.fromstring(original)
            root.find(field).text = value
            ET.ElementTree(root).write(path)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                read_annotation(path, self.images / "crazing_0.jpg")
        root = ET.fromstring(original)
        root.append(deepcopy(root.find("filename")))
        ET.ElementTree(root).write(path)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.build()
        root = ET.fromstring(original)
        root.remove(root.find("object"))
        ET.ElementTree(root).write(path)
        with self.assertRaisesRegex(ValueError, "no objects"):
            self.build()
        path.write_text("<broken>")
        with self.assertRaises(ET.ParseError):
            self.build()

    def test_invalid_images_missing_pairs_and_classes_are_rejected(self) -> None:
        image = self.images / "crazing_0.jpg"
        original = image.read_bytes()
        image.write_bytes(b"broken")
        with self.assertRaises(OSError):
            self.build()
        Image.new("RGB", (300, 200)).save(image)
        with self.assertRaisesRegex(ValueError, "200x200"):
            self.build()
        image.write_bytes(original)
        image.unlink()
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            self.build()
        (self.annotations / "crazing_0.xml").unlink()
        for image in self.images.glob("crazing_*.jpg"):
            (self.annotations / f"{image.stem}.xml").unlink()
            image.unlink()
        with self.assertRaisesRegex(ValueError, "three unique pixel groups"):
            self.build()

    def test_invalid_split_settings_are_rejected(self) -> None:
        for kwargs in (
            {"seed": True},
            {"val_fraction": 0},
            {"test_fraction": 1},
            {"val_fraction": float("nan")},
            {"test_fraction": float("inf")},
            {"val_fraction": 0.6, "test_fraction": 0.4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_snapshot_hash_and_refusal_to_replace_changed_output(self) -> None:
        manifest, report = self.build()
        output = self.data / "snapshot"
        publish_snapshot(output, manifest, report)
        publish_snapshot(output, *self.build())
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
        self.assertEqual(json.loads((output / "report.json").read_text()), saved)


if __name__ == "__main__":
    unittest.main()
