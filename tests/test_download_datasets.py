"""Test downloads and extraction using tiny ZIP files, without network access."""

import hashlib
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.download_datasets import prepare_dataset


class DownloadDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.archive = self.root / "fixture.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("train/ok/image.jpeg", b"ok-pixels")
            archive.writestr("test/defect/image.jpeg", b"bad-pixels")
        self.spec = {
            "name": "fixture",
            "version": 1,
            "url": "https://example.invalid/fixture.zip",
            "archive_bytes": self.archive.stat().st_size,
            "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            "unpacked_bytes": 19,
            "counts": {"train/ok": 1, "test/defect": 1},
        }
        self.data = self.root / "data"

    def update_checksum(self) -> None:
        self.spec["archive_bytes"] = self.archive.stat().st_size
        self.spec["sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_download_then_reuse_cache_offline(self) -> None:
        with patch(
            "scripts.download_datasets.urlopen",
            return_value=io.BytesIO(self.archive.read_bytes()),
        ):
            result = prepare_dataset(self.spec, self.data)
        self.assertEqual((result / "train/ok/image.jpeg").read_bytes(), b"ok-pixels")
        with patch(
            "scripts.download_datasets.urlopen",
            side_effect=AssertionError("Network used"),
        ):
            self.assertEqual(prepare_dataset(self.spec, self.data), result)

    def test_partial_download_is_not_published(self) -> None:
        with patch(
            "scripts.download_datasets.urlopen", return_value=io.BytesIO(b"incomplete")
        ):
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                prepare_dataset(self.spec, self.data)
        self.assertEqual(list((self.data / "downloads").iterdir()), [])
        self.assertFalse((self.data / "raw").exists())

    def test_oversized_download_is_not_published(self) -> None:
        content = self.archive.read_bytes() + b"unexpected"
        with patch(
            "scripts.download_datasets.urlopen", return_value=io.BytesIO(content)
        ):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                prepare_dataset(self.spec, self.data)
        self.assertEqual(list((self.data / "downloads").iterdir()), [])

    def test_bad_checksum_is_rejected_before_extraction(self) -> None:
        content = self.archive.read_bytes().replace(b"ok-pixels", b"no-pixels")
        self.archive.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            prepare_dataset(self.spec, self.data, self.archive)
        self.assertFalse(self.data.exists())

    def test_unsafe_paths_are_rejected(self) -> None:
        for name in (
            "../escape.jpeg",
            "/escape.jpeg",
            "C:/escape.jpeg",
            "train\\escape.jpeg",
        ):
            with self.subTest(name=name):
                with zipfile.ZipFile(self.archive, "w") as archive:
                    archive.writestr(name, b"pixels")
                self.update_checksum()
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    prepare_dataset(self.spec, self.data, self.archive)
                self.assertFalse(self.data.exists())

    def test_symlink_in_archive_is_rejected(self) -> None:
        entry = zipfile.ZipInfo("train/ok/link.jpeg")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(entry, "../../../outside")
        self.update_checksum()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            prepare_dataset(self.spec, self.data, self.archive)

    def test_unexpected_counts_are_rejected(self) -> None:
        self.spec["counts"] = {"train/ok": 2}
        with self.assertRaisesRegex(ValueError, "counts"):
            prepare_dataset(self.spec, self.data, self.archive)
        self.assertFalse(self.data.exists())

    def test_crc_failure_does_not_publish_partial_extraction(self) -> None:
        self.archive.write_bytes(
            self.archive.read_bytes().replace(b"ok-pixels", b"no-pixels")
        )
        self.update_checksum()
        with self.assertRaises(zipfile.BadZipFile):
            prepare_dataset(self.spec, self.data, self.archive)
        self.assertEqual(list((self.data / "raw").iterdir()), [])

    def test_changed_extracted_file_is_not_silently_overwritten(self) -> None:
        result = prepare_dataset(self.spec, self.data, self.archive)
        image = result / "train/ok/image.jpeg"
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "file changed"):
            prepare_dataset(self.spec, self.data, self.archive)
        self.assertEqual(image.read_bytes(), b"changed")

    def add_annotation(self) -> None:
        content = b"<annotation><filename>image.jpeg</filename></annotation>"
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("annotations/image.xml", content)
        self.spec["unpacked_bytes"] += len(content)
        self.spec["counts"]["annotations"] = 1
        self.update_checksum()

    def test_detection_annotations_are_extracted_and_verified_on_rerun(self) -> None:
        self.add_annotation()
        self.spec["file_extensions"] = [".jpeg", ".xml"]
        result = prepare_dataset(self.spec, self.data, self.archive)
        annotation = result / "annotations/image.xml"
        original = annotation.read_bytes()
        self.assertEqual(prepare_dataset(self.spec, self.data, self.archive), result)
        annotation.write_bytes(original.replace(b"image.jpeg", b"other.jpeg"))
        with self.assertRaisesRegex(ValueError, "file changed"):
            prepare_dataset(self.spec, self.data, self.archive)

    def test_classification_dataset_does_not_allow_xml(self) -> None:
        self.add_annotation()
        with self.assertRaisesRegex(ValueError, "Unexpected dataset file"):
            prepare_dataset(self.spec, self.data, self.archive)
        self.assertFalse(self.data.exists())

    def test_detection_dataset_rejects_unlisted_file_types(self) -> None:
        self.spec["file_extensions"] = [".jpeg", ".xml"]
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("annotations/script.py", b"print('unexpected')")
        self.update_checksum()
        with self.assertRaisesRegex(ValueError, "Unexpected dataset file"):
            prepare_dataset(self.spec, self.data, self.archive)
        self.assertFalse(self.data.exists())


if __name__ == "__main__":
    unittest.main()
