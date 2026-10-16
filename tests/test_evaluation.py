"""Check saved-checkpoint evaluation using fixed logits and synthetic images."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import timm
import torch
from torch import nn

from vf_ml.evaluation import evaluate_checkpoint
from vf_ml.models import Classifier, MODEL_NAME
from vf_ml.transforms import ClassificationTransforms


def tiny_model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        samples = []
        for split, count in (("test", 4), ("val", 2)):
            for index in range(count):
                key = f"{split}-{index}.png"
                Image.new("RGB", (20, 20), "white").save(self.root / key)
                samples.append(
                    {
                        "image_key": key,
                        "split": split,
                        "labels": [{"class_index": index % 2, "bbox": None}],
                    }
                )
        samples.append(
            {
                "image_key": "missing-train.png",
                "split": "train",
                "labels": [{"class_index": 0, "bbox": None}],
            }
        )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_type": "classification",
                    "class_map": {"defective": 1, "ok": 0},
                    "samples": samples,
                }
            )
        )
        model = tiny_model()
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.copy_(torch.tensor([1.0, 0.0]))
        self.checkpoint = {
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "epoch": 3,
            "val_loss": 0.5,
            "model_state_dict": model.state_dict(),
            "model": {
                "model_name": MODEL_NAME,
                "timm_version": timm.__version__,
                "class_names": ["ok", "defective"],
                "evaluation_geometry": "full_image_resize",
                "transforms": asdict(ClassificationTransforms(interpolation="bicubic")),
            },
        }
        self.path = self.root / "best.pt"
        torch.save(self.checkpoint, self.path)

    def evaluate(self, split="test", output="report.json", batch_size=3):
        def factory(manifest, *, transforms, pretrained):
            self.assertFalse(pretrained)
            return Classifier(tiny_model(), transforms, {})

        with patch("vf_ml.evaluation.create_classifier", side_effect=factory):
            return evaluate_checkpoint(
                self.path,
                self.manifest,
                self.root,
                self.root / output,
                split=split,
                batch_size=batch_size,
            )

    def test_checkpoint_state_split_and_report_provenance(self) -> None:
        report = self.evaluate()
        self.assertEqual(report["split"], "test")
        self.assertEqual(report["metrics"]["samples"], 4)
        self.assertEqual(report["metrics"]["accuracy"], 0.5)
        self.assertEqual(report["metrics"]["roc_auc"], 0.5)
        self.assertEqual(
            report["metrics"]["confusion_matrix"]["counts"], [[2, 0], [2, 0]]
        )
        expected_loss = torch.nn.functional.cross_entropy(
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([0, 1])
        ).item()
        self.assertAlmostEqual(report["metrics"]["loss"], expected_loss, places=6)
        self.assertEqual(
            report["checkpoint_sha256"],
            hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )
        self.assertEqual(report["manifest_sha256"], self.checkpoint["manifest_sha256"])
        self.assertEqual(report["checkpoint_epoch"], 3)
        self.assertEqual(
            json.loads((self.root / "report.json").read_text())["metrics"],
            report["metrics"],
        )
        with self.assertRaises(FileExistsError):
            self.evaluate()
        self.assertEqual(
            self.evaluate(output="repeat.json", batch_size=1)["metrics"],
            report["metrics"],
        )

    def test_validation_is_labelled_and_does_not_open_test_images(self) -> None:
        for path in self.root.glob("test-*.png"):
            path.unlink()
        report = self.evaluate(split="val")
        self.assertEqual(report["split"], "val")
        self.assertEqual(report["metrics"]["samples"], 2)

    def test_altered_manifest_and_incompatible_metadata_fail_without_a_report(
        self,
    ) -> None:
        mutations = (
            ("class_names", ["defective", "ok"]),
            ("model_name", "other"),
            ("timm_version", "other"),
            ("evaluation_geometry", "center_crop"),
            ("transforms", {}),
        )
        for key, value in mutations:
            bad = deepcopy(self.checkpoint)
            bad["model"][key] = value
            torch.save(bad, self.path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate()
            self.assertFalse((self.root / "report.json").exists())
        torch.save(self.checkpoint, self.path)
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.evaluate()

    def test_state_shape_and_nonfinite_predictions_fail(self) -> None:
        bad = deepcopy(self.checkpoint)
        bad["model_state_dict"]["2.weight"] = torch.zeros(3, 3)
        torch.save(bad, self.path)
        with self.assertRaises(RuntimeError):
            self.evaluate()
        bad = deepcopy(self.checkpoint)
        bad["model_state_dict"]["2.bias"].fill_(float("nan"))
        torch.save(bad, self.path)
        with self.assertRaisesRegex(ValueError, "non-finite logits"):
            self.evaluate()
        self.assertFalse((self.root / "report.json").exists())

    def test_empty_split_and_training_evaluation_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "val or test"):
            self.evaluate(split="train")
        data = json.loads(self.manifest.read_text())
        data["samples"] = [row for row in data["samples"] if row["split"] != "test"]
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "nonempty test"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
