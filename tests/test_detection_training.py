"""Check detector configuration, loader auditing, pinned assets and a real CPU run."""

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from vf_ml.detection_training import (
    DetectionTrainingConfig,
    audit_loader,
    smoke_manifest,
    train_detector,
)
from vf_ml.detector_weights import cached_weight
from vf_ml.yolo import prepare_yolo_dataset


class DetectionTrainingTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        samples = []
        for split, count in (("train", 4), ("val", 2), ("test", 1)):
            for index in range(count):
                key = f"{split}-{index}.png"
                if split != "test":
                    Image.new("RGB", (64, 64), (index * 50, 100, 150)).save(
                        self.root / key
                    )
                samples.append(
                    {
                        "image_key": key,
                        "split": split,
                        "labels": [
                            {"class_index": index % 2, "bbox": [0.1, 0.2, 0.5, 0.6]}
                        ],
                    }
                )
        self.data = {
            "schema_version": 1,
            "task_type": "detection",
            "class_map": {"one": 0, "two": 1},
            "samples": samples,
        }
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.data))

    def test_configuration_and_gpu_guard(self):
        for kwargs in (
            {"epochs": 0},
            {"image_size": 65},
            {"batch_size": True},
            {"seed": 2**32},
            {"workers": -1},
            {"device": "auto"},
            {"fast_dev_run": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DetectionTrainingConfig(**kwargs)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                train_detector(
                    self.manifest,
                    self.root,
                    self.root / "run",
                    DetectionTrainingConfig(device="cuda"),
                    cache_dir=self.root,
                )
        self.assertFalse((self.root / "run").exists())

    def test_smoke_selection_is_repeatable_and_does_not_include_test(self):
        contents = self.manifest.read_bytes()
        selected = smoke_manifest(contents)
        self.assertEqual(smoke_manifest(contents), selected)
        data = json.loads(selected)
        self.assertEqual(data["class_map"], self.data["class_map"])
        self.assertEqual({row["split"] for row in data["samples"]}, {"train", "val"})
        self.assertEqual(len(data["samples"]), 6)
        self.assertEqual(data["samples"], self.data["samples"][:6])

    def test_loader_audit_permits_only_duplicate_rows(self):
        self.data["samples"][0]["labels"] *= 2
        self.manifest.write_text(json.dumps(self.data))
        prepared = prepare_yolo_dataset(self.manifest, self.root, self.root / "dataset")
        labels = []
        for row in prepared["images"]:
            if row["split"] != "train":
                continue
            values = np.loadtxt(
                self.root / "dataset" / row["label_file"], dtype=np.float32, ndmin=2
            )
            values = np.unique(values, axis=0)
            labels.append(
                {
                    "im_file": str(self.root / "dataset" / row["image_file"]),
                    "normalized": True,
                    "bbox_format": "xywh",
                    "cls": values[:, :1],
                    "bboxes": values[:, 1:],
                }
            )

        class Dataset:
            pass

        dataset = Dataset()
        dataset.labels = labels
        result = audit_loader(dataset, prepared, self.root / "dataset", "train")
        self.assertEqual(
            result,
            {
                "images": 4,
                "manifest_boxes": 5,
                "effective_boxes": 4,
                "duplicate_rows_removed": 1,
            },
        )
        dataset.labels = labels[:-1]
        with self.assertRaisesRegex(ValueError, "image set"):
            audit_loader(dataset, prepared, self.root / "dataset", "train")
        dataset.labels = deepcopy(labels)
        dataset.labels[0]["bboxes"][0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "changed labels"):
            audit_loader(dataset, prepared, self.root / "dataset", "train")

    def test_failed_preparation_is_recorded_and_existing_run_refused(self):
        (self.root / "train-0.png").unlink()
        output = self.root / "run"
        with self.assertRaises(FileNotFoundError):
            train_detector(self.manifest, self.root, output, cache_dir=self.root)
        report = json.loads((output / "run.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse((output / "best.pt").exists())
        with self.assertRaises(FileExistsError):
            train_detector(self.manifest, self.root, output, cache_dir=self.root)

    def test_real_cpu_smoke_training_saves_checkpoint_and_val_history(self):
        output = self.root / "run"
        command = [
            sys.executable,
            "-m",
            "scripts.train_detector",
            "--manifest",
            str(self.manifest),
            "--data-dir",
            str(self.root),
            "--output",
            str(output),
            "--cache-dir",
            str(self.root / "cache"),
            "--fast-dev-run",
            "--random-init",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        )
        self.assertEqual(
            result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:]
        )
        run = json.loads((output / "run.json").read_text())
        history = json.loads((output / "history.json").read_text())
        self.assertEqual(run["status"], "completed")
        self.assertTrue(run["smoke_run"])
        self.assertFalse(run["pretrained"])
        self.assertFalse(run["amp_enabled"])
        self.assertEqual(run["effective_config"]["epochs"], 1)
        self.assertEqual(run["effective_config"]["image_size"], 64)
        self.assertEqual(run["selection_split"], "val")
        self.assertEqual(run["class_map"], self.data["class_map"])
        self.assertEqual(run["loader_audit"]["train"]["images"], 4)
        self.assertEqual(run["loader_audit"]["val"]["images"], 2)
        self.assertEqual(len(history), 1)
        self.assertGreater(run["optimizer_steps"], 0)
        self.assertEqual(run["optimizer_steps"], history[0]["optimizer_steps"])
        self.assertEqual(run["skipped_optimizer_steps"], 0)
        self.assertEqual(run["best_epoch"], history[0]["epoch"])
        self.assertEqual(run["best_fitness"], history[0]["fitness"])
        self.assertEqual(
            run["checkpoint_sha256"],
            hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (output / "source-manifest.json").read_bytes(), self.manifest.read_bytes()
        )
        self.assertFalse((output / "dataset/images/test").exists())
        self.assertEqual(
            run["source_manifest_sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            run["manifest_sha256"],
            hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        )
        self.assertTrue((output / "fit/args.yaml").is_file())


class DetectorWeightsTest(unittest.TestCase):
    def test_download_cache_and_corruption_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = b"test fixture; never deserialized"
            spec = {"tiny.pt": (len(payload), hashlib.sha256(payload).hexdigest())}
            with (
                patch("vf_ml.detector_weights.WEIGHTS", spec),
                patch(
                    "vf_ml.detector_weights.urlopen", return_value=io.BytesIO(payload)
                ) as download,
            ):
                path = cached_weight(directory, "tiny.pt")
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(cached_weight(directory, "tiny.pt"), path)
                self.assertEqual(download.call_count, 1)
                path.write_bytes(b"wrong")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    cached_weight(directory, "tiny.pt")
                self.assertEqual(download.call_count, 1)

    def test_invalid_download_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("vf_ml.detector_weights.WEIGHTS", {"tiny.pt": (4, "invalid")}),
                patch(
                    "vf_ml.detector_weights.urlopen", return_value=io.BytesIO(b"wrong")
                ),
            ):
                with self.assertRaises(ValueError):
                    cached_weight(directory, "tiny.pt")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
