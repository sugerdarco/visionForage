"""Check image decoding, split isolation, and reproducible worker batches."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch.utils.data import DataLoader

from vf_ml.classification import ClassificationDataset
from vf_ml.manifest import parse_manifest
from vf_ml.transforms import ClassificationTransforms, classification_transform


class ClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        image = Image.new("RGB", (60, 80))
        image.putdata(
            [(x * 4, y * 3, (x + y) % 256) for y in range(80) for x in range(60)]
        )
        samples = []
        for split in ("train", "val", "test"):
            for index in range(4):
                key = f"{split}-{index}.png"
                image.save(self.root / key)
                samples.append(
                    {
                        "image_key": key,
                        "split": split,
                        "labels": [{"class_index": index % 2, "bbox": None}],
                    }
                )
        self.raw = {
            "schema_version": 1,
            "task_type": "classification",
            "class_map": {"defective": 1, "ok": 0},
            "samples": samples,
        }
        self.manifest = parse_manifest(self.raw)
        self.config = ClassificationTransforms(
            image_size=32, horizontal_flip=True, vertical_flip=True
        )

    def dataset(self, split: str) -> ClassificationDataset:
        return ClassificationDataset(self.manifest, self.root, split, self.config)

    def test_split_labels_keys_and_batch_contract(self) -> None:
        # Unlisted files must never become training samples.
        (self.root / "unlisted.png").write_bytes(b"not an image")
        dataset = self.dataset("val")
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.class_names, ("ok", "defective"))
        batch = next(iter(DataLoader(dataset, batch_size=4)))
        self.assertEqual(batch["image"].shape, (4, 3, 32, 32))
        self.assertEqual(batch["image"].dtype, torch.float32)
        self.assertEqual(batch["image"].device.type, "cpu")
        self.assertTrue(torch.isfinite(batch["image"]).all())
        self.assertEqual(batch["target"].dtype, torch.int64)
        self.assertEqual(batch["target"].tolist(), [0, 1, 0, 1])
        self.assertEqual(batch["image_key"], [f"val-{index}.png" for index in range(4)])

    def test_validation_and_test_match_shared_inference_preprocessing(self) -> None:
        inference = classification_transform(self.config)
        for split in ("val", "test"):
            dataset = self.dataset(split)
            torch.manual_seed(1)
            actual = dataset[0]["image"]
            torch.manual_seed(999)
            self.assertTrue(torch.equal(actual, dataset[0]["image"]))
            with Image.open(self.root / f"{split}-0.png") as image:
                self.assertTrue(torch.equal(actual, inference(image)))

    def test_grayscale_scaling_and_normalization(self) -> None:
        Image.new("L", (32, 32), 128).save(self.root / "val-0.png")
        actual = self.dataset("val")[0]["image"]
        expected = (
            torch.tensor([128 / 255] * 3) - torch.tensor(self.config.mean)
        ) / torch.tensor(self.config.std)
        torch.testing.assert_close(actual, expected[:, None, None].expand(3, 32, 32))

    def test_evaluation_keeps_border_pixels(self) -> None:
        image = Image.new("RGB", (40, 20), "black")
        image.paste("white", (0, 0, 4, 20))
        config = ClassificationTransforms(image_size=20, mean=(0, 0, 0), std=(1, 1, 1))
        actual = classification_transform(config)(image)
        self.assertTrue((actual[:, :, 0] > 0.9).all())
        self.assertTrue((actual[:, :, -1] == 0).all())

    def test_training_randomness_replays_with_torch_seed(self) -> None:
        dataset = self.dataset("train")
        torch.manual_seed(7)
        first = dataset[0]["image"]
        second = dataset[0]["image"]
        self.assertFalse(torch.equal(first, second))
        torch.manual_seed(7)
        self.assertTrue(torch.equal(first, dataset[0]["image"]))

    def test_spawned_workers_replay_batches_with_a_seeded_generator(self) -> None:
        def collect() -> list[dict]:
            return list(
                DataLoader(
                    self.dataset("train"),
                    batch_size=2,
                    shuffle=True,
                    num_workers=2,
                    multiprocessing_context="spawn",
                    generator=torch.Generator().manual_seed(42),
                )
            )

        first, second = collect(), collect()
        self.assertEqual(len(first), 2)
        for left, right in zip(first, second):
            self.assertEqual(left["image_key"], right["image_key"])
            self.assertTrue(torch.equal(left["target"], right["target"]))
            self.assertTrue(torch.equal(left["image"], right["image"]))

    def test_missing_corrupt_and_escaping_images_fail_with_the_key(self) -> None:
        path = self.root / "val-0.png"
        dataset = self.dataset("val")
        path.unlink()
        with self.assertRaisesRegex(OSError, "val-0.png"):
            dataset[0]
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(OSError, "val-0.png"):
            dataset[0]
        path.unlink()
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "image.png"
            Image.new("RGB", (32, 32)).save(target)
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes data_root: val-0.png"):
                dataset[0]

    def test_wrong_task_unknown_or_empty_split_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            self.dataset("validation")
        self.raw["samples"] = self.raw["samples"][:4]
        with self.assertRaisesRegex(ValueError, "no samples"):
            ClassificationDataset(parse_manifest(self.raw), self.root, "val")
        self.raw["task_type"] = "detection"
        for row in self.raw["samples"]:
            row["labels"] = []
        with self.assertRaisesRegex(ValueError, "classification manifest"):
            ClassificationDataset(parse_manifest(self.raw), self.root, "train")

    def test_configuration_round_trip_and_invalid_values(self) -> None:
        self.assertEqual(
            ClassificationTransforms(**json.loads(json.dumps(asdict(self.config)))),
            self.config,
        )
        for settings in (
            {"image_size": 0},
            {"image_size": True},
            {"crop_min_scale": float("nan")},
            {"horizontal_flip": 0.5},
            {"std": [1, 0, 1]},
            {"mean": [0]},
            {"rotation_degrees": float("inf")},
            {"brightness": -1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ClassificationTransforms(**settings)


if __name__ == "__main__":
    unittest.main()
