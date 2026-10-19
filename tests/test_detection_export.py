"""Verify real detector ONNX export, preprocessing, dynamic batches and failure gates."""

import hashlib
import json
import os
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image

import test_detection_evaluation as fixtures
from vf_ml.detection_export import check_predictions, export_detector


class DetectionExportTest(unittest.TestCase):
    save_run = fixtures.DetectionEvaluationTest.save_run

    def setUp(self):
        fixtures.DetectionEvaluationTest.setUp(self)
        self.output = self.root / "onnx"

    def export(self, **kwargs):
        return export_detector(
            self.checkpoint, self.manifest, self.root, self.output, **kwargs
        )

    def test_invalid_inputs_and_missing_samples_never_load_checkpoint(self):
        # The fixture contains invalid pickle bytes; validation must reject earlier.
        for options in (
            {"parity_samples": 1},
            {"parity_samples": True},
            {"cpu_threads": 0},
            {"split": "train"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.export(**options)
        with self.assertRaisesRegex(ValueError, "32 distinct test images"):
            self.export()
        self.assertFalse(self.output.exists())
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Manifest hash"):
            self.export(parity_samples=2)
        self.assertFalse(self.output.exists())

    def test_existing_directories_and_broken_symlinks_are_preserved(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("unchanged")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual((self.output / "keep.txt").read_text(), "unchanged")
        (self.output / "keep.txt").unlink()
        self.output.rmdir()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertTrue(self.output.is_symlink())

    def test_prediction_contract_and_absolute_not_relative_tolerance(self):
        reference = np.zeros((2, 6, 84), dtype=np.float32)
        reference[:, :4] = 100
        reference[:, 4:] = 0.5
        self.assertEqual(
            check_predictions(reference, reference.copy(), 2, 84),
            {"boxes": 0, "scores": 0},
        )
        different = reference.copy()
        different[:, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "ONNX parity failed"):
            check_predictions(reference, different, 2, 84)
        for actual in (
            reference[:, :5],
            reference.transpose(0, 2, 1),
            reference.astype(np.float64),
            reference * np.nan,
        ):
            with self.subTest(shape=actual.shape), self.assertRaises(ValueError):
                check_predictions(reference, actual, 2, 84)
        for channel, value in ((2, -1), (4, -0.01), (4, 1.01)):
            actual = reference.copy()
            actual[:, channel] = value
            with (
                self.subTest(channel=channel, value=value),
                self.assertRaises(ValueError),
            ):
                check_predictions(actual, actual, 2, 84)

    def test_real_yolov8_graph_test_gate_validation_mode_and_failed_parity_cleanup(
        self,
    ):
        # Real architecture with random weights; only synthetic test/val images.
        for split in ("test", "val"):
            for index in range(32 if split == "test" else 2):
                key = f"{split}-{index}.png"
                width, height = (37, 21) if index % 2 == 0 else (29, 71)
                pixels = np.arange(width * height * 3, dtype=np.uint8).reshape(
                    height, width, 3
                )
                pixels = np.roll(pixels, index, axis=0)
                Image.fromarray(pixels).save(self.root / key)
                self.data["samples"].append(
                    {"image_key": key, "split": split, "labels": []}
                )
        # Retain one missing training row to prove split isolation.
        self.data["samples"] = [self.data["samples"][0]] + self.data["samples"][4:]
        self.manifest.write_text(json.dumps(self.data))
        self.run["manifest_sha256"] = hashlib.sha256(
            self.manifest.read_bytes()
        ).hexdigest()
        self.save_run()
        code = r"""
import hashlib, json, sys
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionValidator
from vf_ml.detection_export import export_detector, parity_images
from vf_ml.manifest import load_manifest
from vf_ml.yolo import prepare_yolo_dataset
import vf_ml.detection_export as exporter

root = Path(sys.argv[1])
torch.set_num_threads(2)
torch.manual_seed(42)
model = DetectionModel('yolov8s.yaml', nc=2, verbose=False)
model.names = {0: 'one', 1: 'two'}
torch.save({'model': model, 'train_args': {'task': 'detect'}}, root / 'best.pt')
run = json.loads((root / 'run.json').read_text())
run['checkpoint_sha256'] = hashlib.sha256((root / 'best.pt').read_bytes()).hexdigest()
(root / 'run.json').write_text(json.dumps(run))
args = (root / 'best.pt', root / 'manifest.json', root)
report = export_detector(*args, root / 'onnx')
assert report['parity']['meets_32_test_image_check']
assert report['parity']['batch_sizes'] == [1, 8]

# Compare actual loader pixels to the documented serving contract, including odd
# padding, channel order and resize rounding; compare against validator batches too.
work = root / 'preprocessing'
work.mkdir()
prepared = prepare_yolo_dataset(root / 'manifest.json', root, work / 'dataset', evaluation_split='val')
manifest = load_manifest(root / 'manifest.json')
images, records, _ = parity_images(work, prepared, manifest, 'val', 64, 2, 32)
validator = DetectionValidator(args={'imgsz': 64, 'rect': False, 'half': False, 'workers': 0}, save_dir=work / 'validate')
validator.data = {'names': {0: 'one', 1: 'two'}, 'nc': 2, 'channels': 3}
validator.stride = 32
validator.device = torch.device('cpu')
loader = validator.get_dataloader(str(work / 'dataset/images/val'), 2)
batch = validator.preprocess(next(iter(loader)))
for index, record in enumerate(records):
    raw = cv2.imread(str(root / record['image_key']))
    h, w = raw.shape[:2]
    ratio = 64 / max(h, w)
    rw, rh = min(int(np.ceil(w * ratio)), 64), min(int(np.ceil(h * ratio)), 64)
    raw = cv2.resize(raw, (rw, rh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (64 - rh) / 2, (64 - rw) / 2
    raw = cv2.copyMakeBorder(raw, round(dh-.1), round(dh+.1), round(dw-.1), round(dw+.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    expected = np.ascontiguousarray(raw[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255
    np.testing.assert_array_equal(images[index].numpy(), expected)
    path = str((work / 'dataset' / record['image_file']).resolve())
    position = list(batch['im_file']).index(path)
    torch.testing.assert_close(images[index], batch['img'][position], rtol=0, atol=0)

small = export_detector(*args, root / 'onnx-val', split='val', parity_samples=2)
assert not small['parity']['meets_32_test_image_check']
assert small['parity']['batch_sizes'] == [1, 2]

original = exporter.ort.InferenceSession
def changed_session(*args, **kwargs):
    session = original(*args, **kwargs)
    altered = Mock(wraps=session)
    def run(*a, **kw):
        outputs = session.run(*a, **kw)
        outputs[0][:, :4] += .1
        return outputs
    altered.run.side_effect = run
    return altered
with patch.object(exporter.ort, 'InferenceSession', side_effect=changed_session):
    try:
        export_detector(*args, root / 'failed', split='val', parity_samples=2)
    except ValueError as error:
        assert 'ONNX parity failed' in str(error), str(error)
    else:
        raise AssertionError('Parity disagreement did not fail')
assert not (root / 'failed').exists()
assert not list(root.glob('.detection-onnx-*'))
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.root)],
            capture_output=True,
            text=True,
            timeout=120,
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
        report = json.loads((self.output / "verification.json").read_text())
        metadata = json.loads((self.output / "metadata.json").read_text())
        self.assertEqual(report["parity"]["samples"], 32)
        self.assertEqual(len(set(report["parity"]["image_keys"])), 32)
        self.assertEqual(len(report["parity"]["image_sha256"]), 32)
        self.assertLessEqual(report["parity"]["max_absolute_error"], 1e-3)
        self.assertEqual(report["latency"]["measured_runs"], 200)
        self.assertGreater(report["latency"]["p95_ms"], 0)
        self.assertEqual(metadata["input"]["shape"], ["batch", 3, 64, 64])
        self.assertEqual(metadata["output"]["shape"], ["batch", 6, 84])
        self.assertFalse(metadata["input"]["preprocessing_in_graph"])
        self.assertFalse(metadata["output"]["nms_in_graph"])
        self.assertEqual(metadata["class_map"], self.data["class_map"])
        self.assertEqual(metadata["opset"], 17)
        self.assertEqual(
            metadata["model_sha256"],
            hashlib.sha256((self.output / "model.onnx").read_bytes()).hexdigest(),
        )
        self.assertEqual(report["model_sha256"], metadata["model_sha256"])
        for name, field in (
            ("best.pt", "checkpoint_sha256"),
            ("manifest.json", "manifest_sha256"),
            ("run.json", "run_sha256"),
        ):
            self.assertEqual(
                metadata[field],
                hashlib.sha256((self.root / name).read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
