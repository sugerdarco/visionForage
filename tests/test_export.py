"""Exercise real ONNX export, runtime parity, and bundle publication failures."""

import hashlib
import json
from unittest.mock import Mock, patch
import unittest

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
import torch

import test_evaluation as fixtures
from vf_ml.export import _check_logits, export_checkpoint
from vf_ml.models import Classifier


class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        fixtures.EvaluationTest.setUp(self)
        # Nonzero weights ensure parity exercises the graph's inputs and parameters.
        self.checkpoint["model_state_dict"]["2.weight"] = torch.tensor(
            [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]
        )
        torch.save(self.checkpoint, self.path)

    def export(self, **kwargs):
        def factory(manifest, *, transforms, pretrained):
            self.assertFalse(pretrained)
            return Classifier(fixtures.tiny_model(), transforms, {})

        with patch("vf_ml.checkpoint.create_classifier", side_effect=factory):
            return export_checkpoint(
                self.path, self.manifest, self.root, self.root / "onnx", **kwargs
            )

    def test_real_dynamic_graph_and_full_32_image_gate(self) -> None:
        contents = json.loads(self.manifest.read_text())
        for index in range(4, 32):
            key = f"test-{index}.png"
            Image.new("RGB", (20, 20), (index * 7, 50, 200)).save(self.root / key)
            contents["samples"].append(
                {
                    "image_key": key,
                    "split": "test",
                    "labels": [{"class_index": index % 2, "bbox": None}],
                }
            )
        self.manifest.write_text(json.dumps(contents))
        self.checkpoint["manifest_sha256"] = hashlib.sha256(
            self.manifest.read_bytes()
        ).hexdigest()
        torch.save(self.checkpoint, self.path)
        # Neither training nor validation images should be opened.
        for path in self.root.glob("val-*.png"):
            path.unlink()
        report = self.export()
        self.assertTrue(report["parity"]["meets_32_test_image_check"])
        self.assertEqual(len(set(report["parity"]["image_keys"])), 32)
        self.assertEqual(report["parity"]["batch_sizes"], [1, 8])
        self.assertLessEqual(report["parity"]["max_absolute_error"], 1e-3)
        self.assertEqual(report["latency"]["measured_runs"], 200)
        self.assertGreater(report["latency"]["p95_ms"], 0)
        bundle = self.root / "onnx"
        model = onnx.load(bundle / "model.onnx")
        self.assertEqual(model.opset_import[0].version, 17)
        metadata = json.loads((bundle / "metadata.json").read_text())
        self.assertEqual(metadata["class_map"], {"ok": 0, "defective": 1})
        self.assertFalse(metadata["input"]["preprocessing_in_graph"])
        self.assertEqual(
            metadata["checkpoint_sha256"],
            hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            metadata["model_sha256"],
            hashlib.sha256((bundle / "model.onnx").read_bytes()).hexdigest(),
        )
        self.assertEqual(json.loads((bundle / "verification.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(json.loads((bundle / "verification.json").read_text()), report)

    def test_small_validation_check_is_explicit(self) -> None:
        for path in self.root.glob("test-*.png"):
            path.unlink()
        report = self.export(split="val", parity_samples=2)
        self.assertEqual(report["parity"]["split"], "val")
        self.assertFalse(report["parity"]["meets_32_test_image_check"])
        self.assertEqual(report["parity"]["batch_sizes"], [1, 2])

    def test_too_few_samples_and_changed_manifest_publish_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 distinct test images"):
            self.export()
        self.assertFalse((self.root / "onnx").exists())
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.export(parity_samples=4)
        self.assertFalse((self.root / "onnx").exists())

    def test_runtime_disagreement_removes_staged_artifacts(self) -> None:
        original = ort.InferenceSession

        def changed_runtime(*args, **kwargs):
            session = original(*args, **kwargs)
            changed = Mock(wraps=session)
            changed.run.side_effect = lambda *a, **kw: [session.run(*a, **kw)[0] + 0.1]
            return changed

        with patch("vf_ml.export.ort.InferenceSession", side_effect=changed_runtime):
            with self.assertRaisesRegex(ValueError, "ONNX parity failed"):
                self.export(parity_samples=4)
        self.assertFalse((self.root / "onnx").exists())
        self.assertEqual(list(self.root.glob(".onnx-*")), [])

    def test_logit_check_uses_absolute_error_and_rejects_invalid_output(self) -> None:
        reference = np.array([[100.0, 0.0]], dtype=np.float32)
        self.assertEqual(_check_logits(reference, reference.copy(), 2), 0.0)
        for actual in (
            reference + 0.01,
            np.array([[float("nan"), 0]], dtype=np.float32),
            reference[:, :1],
            reference.astype(np.float64),
        ):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                _check_logits(reference, actual, 2)

    def test_invalid_configuration_does_not_create_output(self) -> None:
        for kwargs in (
            {"parity_samples": 1},
            {"parity_samples": True},
            {"cpu_threads": 0},
            {"split": "train"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.export(**kwargs)
        self.assertFalse((self.root / "onnx").exists())


if __name__ == "__main__":
    unittest.main()
