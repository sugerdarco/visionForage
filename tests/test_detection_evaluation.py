"""Check detector provenance, split isolation and threshold-specific box matching."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from vf_ml.detection_evaluation import (
    _validator_class,
    detection_metrics,
    evaluate_detector,
    read_detection_run,
)
from vf_ml.detection_training import ULTRALYTICS_VERSION
from vf_ml.yolo import prepare_yolo_dataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DetectionEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"not a pickle; provenance checks must run first")
        self.manifest = self.root / "manifest.json"
        self.data = {
            "schema_version": 1,
            "task_type": "detection",
            "class_map": {"one": 0, "two": 1},
            "samples": [
                {"image_key": "missing-train.png", "split": "train", "labels": []},
                {"image_key": "missing-val.png", "split": "val", "labels": []},
                {
                    "image_key": "test.png",
                    "split": "test",
                    "labels": [
                        {"class_index": 0, "bbox": [0.1, 0.2, 0.5, 0.6]},
                        {"class_index": 0, "bbox": [0.1, 0.2, 0.5, 0.6]},
                    ],
                },
                {"image_key": "background.png", "split": "test", "labels": []},
            ],
        }
        self.manifest.write_text(json.dumps(self.data))
        for name in ("test.png", "background.png"):
            Image.new("RGB", (80, 40), (30, 70, 110)).save(self.root / name)
        self.run_path = self.root / "run.json"
        self.run = {
            "schema_version": 1,
            "status": "completed",
            "task_type": "detection",
            "model_name": "yolov8s",
            "ultralytics_version": ULTRALYTICS_VERSION,
            "selection_split": "val",
            "selection_metric": "validation mAP@0.5:0.95",
            "manifest_sha256": digest(self.manifest),
            "checkpoint_sha256": digest(self.checkpoint),
            "class_map": self.data["class_map"],
            "effective_config": {"image_size": 64},
            "best_epoch": 1,
            "smoke_run": True,
        }
        self.save_run()
        self.output = self.root / "report.json"

    def save_run(self):
        self.run_path.write_text(json.dumps(self.run))

    def read(self, split="test"):
        return read_detection_run(self.checkpoint, self.manifest, self.run_path, split)

    def test_provenance_rejects_changed_files_and_incompatible_runs(self):
        self.assertEqual(self.read()[2], self.run)
        original = deepcopy(self.run)
        for key, value in (
            ("status", "failed"),
            ("manifest_sha256", "wrong"),
            ("checkpoint_sha256", "wrong"),
            ("class_map", {"two": 0, "one": 1}),
            ("ultralytics_version", "unknown"),
            ("selection_split", "test"),
            ("best_epoch", True),
            ("effective_config", {"image_size": 65}),
            ("smoke_run", "false"),
        ):
            self.run = {**original, key: value}
            self.save_run()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read()
        self.run = original
        self.save_run()
        self.checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
            self.read()
        self.assertFalse(self.output.exists())

    def test_missing_split_and_invalid_options_fail_before_loading_pickle(self):
        self.data["samples"] = self.data["samples"][:2]
        self.manifest.write_text(json.dumps(self.data))
        self.run["manifest_sha256"] = digest(self.manifest)
        self.save_run()
        with self.assertRaisesRegex(ValueError, "no samples for test"):
            self.read()
        for options in (
            {"split": "train"},
            {"device": "auto"},
            {"batch_size": True},
            {"cpu_threads": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_detector(
                    self.checkpoint, self.manifest, self.root, self.output, **options
                )
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluate_detector(
                    self.checkpoint,
                    self.manifest,
                    self.root,
                    self.output,
                    device="cuda",
                )
        self.output.write_text("keep me")
        with self.assertRaises(FileExistsError):
            evaluate_detector(self.checkpoint, self.manifest, self.root, self.output)
        self.assertEqual(self.output.read_text(), "keep me")

    def test_evaluation_layout_reads_only_requested_images(self):
        report = prepare_yolo_dataset(
            self.manifest, self.root, self.root / "dataset", evaluation_split="test"
        )
        self.assertEqual(report["included_splits"], ["test"])
        self.assertTrue(report["evaluation_only"])
        self.assertEqual(report["counts"], {"test": {"images": 2, "boxes": 2}})
        self.assertFalse((self.root / "dataset/images/train").exists())
        self.assertFalse((self.root / "dataset/images/val").exists())
        self.assertEqual(
            (self.root / "dataset/manifest.json").read_bytes(),
            self.manifest.read_bytes(),
        )

    def test_real_checkpoint_test_evaluation_in_fresh_process(self):
        # Locally generated random weights: no downloads, no training or held-out NEU data.
        code = """
import hashlib, json, sys
from pathlib import Path
from unittest.mock import patch
import torch
from ultralytics.nn.tasks import DetectionModel
from vf_ml.detection_evaluation import evaluate_detector
root = Path(sys.argv[1])
torch.set_num_threads(2)
model = DetectionModel('yolov8s.yaml', nc=2, verbose=False)
model.names = {0: 'one', 1: 'two'}
torch.save({'model': model, 'train_args': {'task': 'detect'}}, root / 'best.pt')
run = json.loads((root / 'run.json').read_text())
run['checkpoint_sha256'] = hashlib.sha256((root / 'best.pt').read_bytes()).hexdigest()
(root / 'run.json').write_text(json.dumps(run))
# The pinned validator checks/downloads fonts even with plots=False.
# This headless test exercises real inference, not optional plotting assets.
with patch('ultralytics.data.utils.check_font', return_value=None):
    evaluate_detector(root / 'best.pt', root / 'manifest.json', root, root / 'report.json', batch_size=2)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.root)],
            capture_output=True,
            text=True,
            timeout=90,
            env={
                **os.environ,
                "YOLO_CONFIG_DIR": str(self.root / "settings"),
                "YOLO_OFFLINE": "true",
                "YOLO_AUTOINSTALL": "false",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["split"], "test")
        self.assertTrue(report["smoke_run"])
        self.assertEqual(
            report["loader_audit"],
            {
                "images": 2,
                "manifest_boxes": 2,
                "effective_boxes": 1,
                "duplicate_rows_removed": 1,
            },
        )
        self.assertEqual(report["metrics"]["fixed_threshold_micro"]["support"], 1)
        self.assertEqual(report["metrics"]["per_class"][0]["ap50"], 0)
        self.assertIsNone(report["metrics"]["per_class"][1]["ap50"])
        self.assertEqual(report["checkpoint_sha256"], digest(self.checkpoint))
        self.assertEqual(report["run_sha256"], digest(self.run_path))
        self.assertEqual(list(self.root.glob(".detection-evaluation-*")), [])


class DetectionMatchingTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        with patch.dict(
            os.environ, {"YOLO_CONFIG_DIR": self.work.name, "YOLO_OFFLINE": "true"}
        ):
            self.validator = _validator_class()(
                save_dir=Path(self.work.name), args={"plots": False}
            )
        self.validator.data = {"val": "fixture", "names": ["one", "two", "absent"]}
        self.validator.init_metrics(
            SimpleNamespace(names={0: "one", 1: "two", 2: "absent"})
        )
        self.batch = {
            "img": torch.zeros(2, 3, 100, 100),
            "batch_idx": torch.tensor([0, 0]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]),
            "ori_shape": [(100, 100)] * 2,
            "ratio_pad": [((1, 1), (0, 0))] * 2,
            "im_file": ["target.png", "background.png"],
        }

    def prediction(self, boxes, scores, classes):
        return {
            "bboxes": torch.tensor(boxes).reshape(-1, 4).float(),
            "conf": torch.tensor(scores),
            "cls": torch.tensor(classes).float(),
            "extra": torch.empty(len(scores), 0),
        }

    def test_fixed_threshold_rematches_and_counts_background_and_wrong_class(self):
        predictions = [
            # Below-threshold perfect box must not steal the retained box's target.
            self.prediction(
                [[30, 30, 70, 70], [31, 31, 70, 70], [5, 5, 15, 15]],
                [0.1, 0.25, 0.8],
                [0, 0, 0],
            ),
            self.prediction([[10, 10, 30, 30]], [0.9], [2]),
        ]
        self.validator.update_metrics(predictions, self.batch)
        np.testing.assert_array_equal(
            self.validator.fixed_counts, [[1, 1, 0], [0, 0, 1], [0, 1, 0]]
        )
        self.validator.get_stats()
        report = detection_metrics(self.validator, ["one", "two", "absent"])
        self.assertAlmostEqual(report["fixed_threshold_micro"]["precision"], 1 / 3)
        self.assertEqual(report["fixed_threshold_micro"]["recall"], 0.5)
        self.assertGreater(report["per_class"][0]["ap50"], 0)
        self.assertEqual(report["per_class"][1]["ap50"], 0)
        self.assertIsNone(report["per_class"][2]["ap50"])
        self.assertAlmostEqual(report["map50"], report["per_class"][0]["ap50"] / 2)

    def test_no_predictions_and_all_background_are_well_defined(self):
        empty = self.prediction([], [], [])
        self.validator.update_metrics([empty, empty], self.batch)
        self.validator.get_stats()
        report = detection_metrics(self.validator, ["one", "two", "absent"])
        self.assertEqual(report["map50"], 0)
        self.assertEqual(report["fixed_threshold_micro"]["false_negatives"], 2)
        self.validator.init_metrics(
            SimpleNamespace(names={0: "one", 1: "two", 2: "absent"})
        )
        self.batch["batch_idx"] = torch.empty(0)
        self.batch["cls"] = torch.empty(0, 1)
        self.batch["bboxes"] = torch.empty(0, 4)
        self.validator.update_metrics([empty, empty], self.batch)
        self.validator.get_stats()
        report = detection_metrics(self.validator, ["one", "two", "absent"])
        self.assertIsNone(report["map50"])
        self.assertEqual(report["fixed_threshold_micro"]["recall"], 0)

    def test_nonfinite_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.validator.postprocess((torch.tensor([float("nan")]), []))

    def test_perfect_boxes_have_expected_interpolated_ap(self):
        self.validator.update_metrics(
            [
                self.prediction([[30, 30, 70, 70], [5, 5, 15, 15]], [0.9, 0.8], [0, 1]),
                self.prediction([], [], []),
            ],
            self.batch,
        )
        self.validator.get_stats()
        report = detection_metrics(self.validator, ["one", "two", "absent"])
        self.assertAlmostEqual(report["map50"], 0.995)
        self.assertAlmostEqual(report["map50_95"], 0.995)
        self.assertEqual(report["fixed_threshold_micro"]["precision"], 1)
        self.assertEqual(report["fixed_threshold_micro"]["recall"], 1)


if __name__ == "__main__":
    unittest.main()
