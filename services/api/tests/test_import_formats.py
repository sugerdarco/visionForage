"""Archive format adapters reject ambiguous data before any database writes."""

from io import BytesIO
import json
import stat
import struct
import unittest
from unittest.mock import patch
import warnings
from zipfile import BadZipFile, ZipFile, ZipInfo, ZIP_DEFLATED
import zlib

import yaml

from import_fixture import coco_document, dataset_files, opened, picture, zipped
from vf_api.import_archive import Archive
from vf_api.import_formats import Label
from vf_api.import_schemas import ImportClaim


class ImportFormatTest(unittest.TestCase):
    def assert_invalid(self, files, format="imagefolder", **kwargs):
        with self.assertRaises((ValueError, yaml.YAMLError, RecursionError)):
            with opened(zipped(files), format, **kwargs):
                self.fail("Invalid archive was accepted")

    def test_imagefolder_sorted_classes_nested_paths_splits_and_explicit_root(self):
        files = {
            "bundle/train/ok/sub/a.png": picture(1),
            "bundle/val/defect/b.png": picture(2),
        }
        with opened(zipped(files), root="bundle") as result:
            self.assertEqual(result.names, ["defect", "ok"])
            self.assertEqual(result.samples[0].labels, [Label("ok")])
            self.assertEqual(
                (result.samples[0].width, result.samples[0].height), (100, 50)
            )
        self.assert_invalid({"a.png": picture()})
        self.assert_invalid({" ok/a.png": picture(), "ok/b.png": picture(1)})

    def test_yolo_conversion_mapping_missing_empty_and_duplicate_boxes(self):
        files = dataset_files("yolo")
        files["data.yaml"] = (
            "names: {0: scratch}\nnc: 1\npath: /unused\ndownload: never_execute\n"
        )
        files["labels/a.txt"] *= 2
        files["images/train/b.png"] = picture(1)
        files["images/val/c.png"] = picture(2)
        files["labels/val/c.txt"] = "\n"
        with opened(zipped(files), "yolo") as result:
            self.assertEqual(result.names, ["scratch"])
            self.assertEqual(len(result.samples[0].labels), 2)
            for actual, expected in zip(
                result.samples[0].labels[0].bbox, (0.1, 0.1, 0.4, 0.4)
            ):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual([row.labels for row in result.samples[1:]], [[], []])

    def test_coco_sparse_ids_pixel_conversion_and_negative_images(self):
        document = coco_document()
        document["images"].append(
            {"id": 99, "file_name": "images/b.png", "width": 100, "height": 50}
        )
        files = {
            "annotations/instances.json": json.dumps(document),
            "images/a.png": picture(),
            "images/b.png": picture(1),
        }
        with opened(
            zipped(files), "coco", annotations_file="annotations/instances.json"
        ) as result:
            self.assertEqual(
                result.samples[0].labels, [Label("scratch", (0.1, 0.1, 0.4, 0.4))]
            )
            self.assertEqual(result.samples[1].labels, [])

    def test_yolo_invalid_maps_labels_orphans_and_ambiguous_stems(self):
        for metadata in (
            "names: [a, a]",
            "names: {1: a}",
            "names: {0: a, 0: b}",
            "names: [a]\nnc: 2",
            "names: [1]",
            "names: true",
            "names: &n [a]\ncopy: *n",
            "names: !!python/object:bad {}",
        ):
            with self.subTest(metadata=metadata):
                self.assert_invalid(
                    {**dataset_files("yolo"), "data.yaml": metadata}, "yolo"
                )
        for line in (
            "0 0.5 0.5 0.4",
            "1 0.5 0.5 0.4 0.4",
            "0.0 0.5 0.5 0.4 0.4",
            "0 0.1 0.1 0.5 0.5",
            "0 nan 0.5 0.4 0.4",
            "0 inf 0.5 0.4 0.4",
            "0 0.5 0.5 -0.1 0.4",
            "0 0.5 0.5 0.4 0.4 0.1",
        ):
            self.assert_invalid({**dataset_files("yolo"), "labels/a.txt": line}, "yolo")
        self.assert_invalid({**dataset_files("yolo"), "labels/orphan.txt": ""}, "yolo")
        self.assert_invalid(
            {**dataset_files("yolo"), "images/a.jpg": picture()}, "yolo"
        )

    def test_coco_rejects_bad_references_crowds_dimensions_and_duplicate_ids(self):
        for change in (
            lambda d: d["annotations"][0].update(image_id=999),
            lambda d: d["annotations"][0].update(category_id=999),
            lambda d: d["annotations"][0].update(iscrowd=1),
            lambda d: d["annotations"][0].update(ignore=1),
            lambda d: d["annotations"][0].update(bbox=[0, 0, 200, 10]),
            lambda d: d["annotations"][0].update(bbox=[True, 0, 10, 10]),
            lambda d: d["images"][0].update(width=101),
            lambda d: d["images"][0].update(file_name="../a.png"),
            lambda d: d["images"][0].update(id=True),
            lambda d: d["images"].append(d["images"][0].copy()),
            lambda d: d["annotations"].append(d["annotations"][0].copy()),
            lambda d: d["categories"].append(d["categories"][0].copy()),
        ):
            document = coco_document()
            change(document)
            self.assert_invalid(
                {"annotations.json": json.dumps(document), "images/a.png": picture()},
                "coco",
            )
        self.assert_invalid({**dataset_files("coco"), "a.png": picture(1)}, "coco")
        for metadata in ('{"images": [], "images": []}', '{"images": NaN}', "[]"):
            self.assert_invalid(
                {**dataset_files("coco"), "annotations.json": metadata}, "coco"
            )

    def test_byte_duplicates_require_identical_labels(self):
        files = {"ok/a.png": picture(), "ok/copy.png": picture()}
        with opened(zipped(files)) as result:
            self.assertEqual(result.samples[0].sha256, result.samples[1].sha256)
        self.assert_invalid({**files, "defect/b.png": picture()})

    def test_bad_paths_links_special_files_duplicate_members_and_directory_counts(self):
        for path in (
            "../escape.png",
            "/absolute.png",
            "x/../a.png",
            "x\\a.png",
            "C:/a.png",
            "x//a.png",
            "x/./a.png",
        ):
            self.assert_invalid({path: picture()})
        for kind in (stat.S_IFLNK, stat.S_IFIFO):
            output = BytesIO()
            with ZipFile(output, "w") as archive:
                member = ZipInfo("ok/a.png")
                member.create_system = 3
                member.external_attr = (kind | 0o777) << 16
                archive.writestr(member, picture())
            with self.assertRaises(ValueError):
                Archive(BytesIO(output.getvalue()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            output = BytesIO()
            with ZipFile(output, "w") as archive:
                archive.writestr("ok/a.png", picture())
                archive.writestr("ok/a.png", picture())
        with self.assertRaises(ValueError):
            Archive(BytesIO(output.getvalue()))
        damaged = bytearray(zipped(dataset_files("imagefolder")))
        end = damaged.rfind(b"PK\x05\x06")
        struct.pack_into("<HH", damaged, end + 8, 0, 0)
        with self.assertRaises(ValueError):
            Archive(BytesIO(damaged))
        self.assert_invalid({"ok": "file", "ok/a.png": picture()})

    def test_limits_bad_images_unknown_members_and_corrupt_zip(self):
        self.assert_invalid({"ok/a.png": b"bad pixels"})
        self.assert_invalid({"ok/a.jpg": picture()})
        self.assert_invalid(
            {"ok/a.png": picture(), "run.py": "print('never executed')"}
        )
        self.assert_invalid({"ok/a.png": picture(), "bomb.txt": "a" * 100_000})
        for target, value in (
            ("vf_api.import_archive.MAX_ENTRIES", 0),
            ("vf_api.import_archive.MAX_EXPANDED_BYTES", 10),
            ("vf_api.import_formats.MAX_TOTAL_PIXELS", 10),
            ("vf_api.import_formats.MAX_LABELS", 0),
        ):
            with patch(target, value):
                self.assert_invalid(dataset_files("imagefolder"))
        with self.assertRaises(ValueError):
            Archive(BytesIO(b"not a zip"))
        with self.assertRaises(ValueError):
            ImportClaim(format="yolo", size=64 * 1024 * 1024 + 1, sha256="a" * 64)

    def test_metadata_and_per_image_box_limits(self):
        files = dataset_files("yolo")
        files["labels/a.txt"] = "0 0.5 0.5 0.4 0.4\n" * 1001
        # Store rather than deflate so this exercises the box limit, not ratio.
        output = BytesIO()
        with ZipFile(output, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        with self.assertRaisesRegex(ValueError, "1,000 boxes"):
            with opened(output.getvalue(), "yolo"):
                pass
        with patch("vf_api.import_archive.MAX_UPLOAD_BYTES", 10):
            self.assert_invalid(dataset_files("yolo"), "yolo")
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("ok/a.png", picture())
        data = bytearray(output.getvalue())
        data[30 + len("ok/a.png")] ^= 0xFF
        with self.assertRaises((BadZipFile, zlib.error)):
            with opened(bytes(data)):
                pass
