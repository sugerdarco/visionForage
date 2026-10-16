"""Verify the classifier contract and pretrained loading without network access."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import timm
import torch

from vf_ml.manifest import parse_manifest
from vf_ml.models import MODEL_NAME, WEIGHT_REVISION, create_classifier
from vf_ml.transforms import ClassificationTransforms


def manifest(task: str = "classification", classes: int = 2):
    return parse_manifest(
        {
            "schema_version": 1,
            "task_type": task,
            "class_map": {f"class-{i}": i for i in reversed(range(classes))},
            "samples": [
                {
                    "image_key": "image.png",
                    "split": "train",
                    "labels": [{"class_index": 0, "bbox": None}]
                    if task == "classification"
                    else [],
                }
            ],
        }
    )


class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_offline_model_has_manifest_ordered_logits_and_trainable_backbone(
        self,
    ) -> None:
        with patch(
            "timm.models._hub.hf_hub_download",
            side_effect=AssertionError("Network used"),
        ):
            classifier = create_classifier(manifest(classes=3), pretrained=False)
        self.assertEqual(
            classifier.metadata["class_names"], ["class-0", "class-1", "class-2"]
        )
        self.assertIsNone(classifier.metadata["weights"])
        json.dumps(classifier.metadata, allow_nan=False)
        self.assertEqual(classifier.transforms.interpolation, "bicubic")
        output = classifier.model(torch.randn(2, 3, 224, 224))
        self.assertEqual(output.shape, (2, 3))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 2])).backward()
        for layer in (classifier.model.conv1, classifier.model.get_classifier()):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_pretrained_backbone_is_loaded_but_imagenet_head_is_replaced(self) -> None:
        source = timm.create_model(MODEL_NAME, pretrained=False)
        with torch.no_grad():
            source.conv1.weight.fill_(0.0123)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_file(source.state_dict(), checkpoint)
            with patch(
                "timm.models._hub.hf_hub_download", return_value=str(checkpoint)
            ) as download:
                classifier = create_classifier(manifest(), cache_dir=directory)
            self.assertEqual(download.call_args.kwargs["revision"], WEIGHT_REVISION)
            self.assertEqual(download.call_args.kwargs["filename"], "model.safetensors")
            self.assertEqual(classifier.model.get_classifier().out_features, 2)
            torch.testing.assert_close(
                classifier.model.conv1.weight, source.conv1.weight
            )
            self.assertEqual(
                classifier.metadata["weights"]["revision"], WEIGHT_REVISION
            )

    def test_pretrained_download_failure_never_falls_back_to_random_weights(
        self,
    ) -> None:
        with patch("timm.models._hub.hf_hub_download", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                create_classifier(manifest())

    def test_wrong_task_and_preprocessing_fail_before_loading_weights(self) -> None:
        with patch("vf_ml.models.timm.create_model") as create:
            for invalid in (manifest(task="detection"), manifest(classes=1)):
                with self.assertRaises(ValueError):
                    create_classifier(invalid)
            config = ClassificationTransforms(interpolation="bicubic")
            for changes in (
                {"image_size": 256},
                {"interpolation": "bilinear"},
                {"mean": (0, 0, 0)},
                {"std": (1, 1, 1)},
            ):
                with (
                    self.subTest(changes=changes),
                    self.assertRaisesRegex(ValueError, "does not match"),
                ):
                    create_classifier(manifest(), transforms=replace(config, **changes))
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
