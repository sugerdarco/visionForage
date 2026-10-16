"""Exercise actual optimization and checkpoint selection with a tiny CPU model."""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from vf_ml.models import Classifier
from vf_ml.training import TrainingConfig, learning_rate_at, run_epoch, train_classifier
from vf_ml.transforms import ClassificationTransforms


def tiny_classifier(manifest, **kwargs):
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    return Classifier(
        model,
        ClassificationTransforms(image_size=16),
        {"class_names": list(manifest.class_names)},
    )


class TrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.previous_threads = torch.get_num_threads()
        self.previous_determinism = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.addCleanup(torch.use_deterministic_algorithms, self.previous_determinism)
        samples = []
        for split in ("train", "val"):
            for index in range(4):
                key = f"{split}-{index}.png"
                Image.new("RGB", (20, 20), "white" if index % 2 else "black").save(
                    self.root / key
                )
                samples.append(
                    {
                        "image_key": key,
                        "split": split,
                        "labels": [{"class_index": index % 2, "bbox": None}],
                    }
                )
        # This missing test file makes accidental test-set access fail the run.
        samples.append(
            {
                "image_key": "test-must-not-be-read.png",
                "split": "test",
                "labels": [{"class_index": 0, "bbox": None}],
            }
        )
        self.manifest = self.root / "input.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_type": "classification",
                    "class_map": {"ok": 0, "defective": 1},
                    "samples": samples,
                }
            )
        )
        self.config = TrainingConfig(epochs=2, batch_size=2)

    def train(self, name="run", config=None):
        with (
            patch("vf_ml.training.create_classifier", side_effect=tiny_classifier),
            redirect_stdout(io.StringIO()),
        ):
            return train_classifier(
                self.manifest,
                self.root,
                self.root / name,
                config or self.config,
                pretrained=False,
            )

    def checkpoint(self, name="run"):
        return torch.load(
            self.root / name / "best.pt", weights_only=True, map_location="cpu"
        )

    def test_real_training_is_reproducible_and_saves_loadable_best_weights(
        self,
    ) -> None:
        first = self.train()
        second = self.train("repeat")
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["epochs_completed"], 2)
        self.assertEqual(first["precision"], "float32")
        self.assertEqual(
            first["manifest_sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (self.root / "run/manifest.json").read_bytes(), self.manifest.read_bytes()
        )
        saved, repeated = self.checkpoint(), self.checkpoint("repeat")
        for key, value in saved["model_state_dict"].items():
            self.assertTrue(torch.equal(value, repeated["model_state_dict"][key]))
        torch.manual_seed(self.config.seed)
        model = tiny_classifier(SimpleNamespace(class_names=("ok", "defective"))).model
        self.assertTrue(
            any(
                not torch.equal(value, saved["model_state_dict"][key])
                for key, value in model.state_dict().items()
            )
        )
        model.load_state_dict(saved["model_state_dict"], strict=True)
        self.assertTrue(torch.isfinite(model(torch.ones(1, 3, 16, 16))).all())
        history = json.loads((self.root / "run/history.json").read_text())
        self.assertEqual(saved["val_loss"], min(row["val"]["loss"] for row in history))
        self.assertEqual(set(history[0]["val"]["recall"]), {"ok", "defective"})
        self.assertEqual(history[0]["val"]["samples"], 4)
        self.assertNotIn("test", history[0])
        with self.assertRaises(FileExistsError):
            self.train()

    def test_early_stopping_retains_best_validation_epoch_not_best_training_epoch(
        self,
    ) -> None:
        metrics = [{"loss": value} for value in (0.9, 0.4, 0.2, 0.5, 0.1, 0.6)]
        with patch("vf_ml.training.run_epoch", side_effect=metrics) as epoch:
            run = self.train(config=replace(self.config, epochs=10, patience=2))
        self.assertEqual(epoch.call_count, 6)
        self.assertEqual(run["stop_reason"], "early_stopping")
        self.assertEqual(run["epochs_completed"], 3)
        self.assertEqual(run["best_epoch"], 1)
        self.assertEqual(self.checkpoint()["epoch"], 1)

    def test_corrupt_training_input_marks_run_failed(self) -> None:
        # Corrupt every training file so weighted sampling cannot avoid the failure.
        for path in self.root.glob("train-*.png"):
            path.write_bytes(b"broken")
        with self.assertRaises(OSError):
            self.train()
        self.assertEqual(
            json.loads((self.root / "run/run.json").read_text())["status"], "failed"
        )
        self.assertFalse((self.root / "run/best.pt").exists())

    def test_nonfinite_loss_stops_without_publishing_a_checkpoint(self) -> None:
        def broken_model(manifest, **kwargs):
            classifier = tiny_classifier(manifest)
            with torch.no_grad():
                classifier.model[-1].weight.fill_(float("nan"))
            return classifier

        with patch("vf_ml.training.create_classifier", side_effect=broken_model):
            with self.assertRaisesRegex(ValueError, "Non-finite loss"):
                train_classifier(
                    self.manifest, self.root, self.root / "broken", self.config
                )
        self.assertFalse((self.root / "broken/best.pt").exists())

    def test_missing_validation_and_training_classes_fail_before_model_creation(
        self,
    ) -> None:
        original = json.loads(self.manifest.read_text())
        for name, rows in (
            ("no-val", [row for row in original["samples"] if row["split"] != "val"]),
            (
                "no-class",
                [
                    row
                    for row in original["samples"]
                    if row["split"] != "train" or row["labels"][0]["class_index"] == 0
                ],
            ),
        ):
            self.manifest.write_text(json.dumps({**original, "samples": rows}))
            with (
                patch("vf_ml.training.create_classifier") as factory,
                self.assertRaises(ValueError),
            ):
                train_classifier(
                    self.manifest, self.root, self.root / name, self.config
                )
            factory.assert_not_called()

    def test_warmup_and_cosine_schedule(self) -> None:
        config = TrainingConfig(epochs=3, learning_rate=0.01)
        rates = [learning_rate_at(step, 4, config) for step in range(12)]
        self.assertEqual(rates[:4], [0.0025, 0.005, 0.0075, 0.01])
        self.assertEqual(rates[4], 0.01)
        self.assertTrue(all(left > right for left, right in zip(rates[4:], rates[5:])))

    def test_validation_metrics_weight_partial_batches_by_sample_count(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        targets = torch.tensor([0, 1, 1])
        loader = DataLoader(
            [
                {"image": image, "target": target}
                for image, target in zip(logits, targets)
            ],
            batch_size=2,
        )
        result = run_epoch(nn.Identity(), loader, "cpu", ("ok", "defective"))
        self.assertAlmostEqual(
            result["loss"],
            torch.nn.functional.cross_entropy(logits, targets).item(),
            places=6,
        )
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["recall"], {"ok": 1.0, "defective": 0.5})

    def test_autocast_training_path_on_cpu(self) -> None:
        model = nn.Linear(3, 2)
        loader = DataLoader(
            [
                {"image": torch.ones(3), "target": 0},
                {"image": torch.zeros(3), "target": 1},
            ],
            batch_size=2,
        )
        metrics = run_epoch(
            model,
            loader,
            "cpu",
            ("ok", "defective"),
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            amp_dtype=torch.bfloat16,
            config=self.config,
        )
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_invalid_configuration_is_rejected(self) -> None:
        for changes in (
            {"epochs": 0},
            {"workers": -1},
            {"batch_size": True},
            {"seed": 2**32},
            {"learning_rate": float("nan")},
            {"label_smoothing": 1},
            {"device": "auto"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig(**changes)


if __name__ == "__main__":
    unittest.main()
