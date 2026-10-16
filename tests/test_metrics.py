"""Check metrics against hand calculations and pairwise AUC definitions."""

import json
import unittest

import numpy as np

from vf_ml.metrics import _binary_auc, classification_metrics


class MetricsTest(unittest.TestCase):
    def score(self, targets, scores):
        scores = np.array(scores)
        return classification_metrics(
            np.array(targets),
            np.column_stack((1 - scores, scores)),
            [f"image-{i}" for i in range(len(targets))],
            ("ok", "defective"),
        )

    def test_binary_metrics_match_hand_calculation(self) -> None:
        result = self.score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"]["counts"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(result["per_class"]["ok"]["precision"], 2 / 3)
        self.assertEqual(result["per_class"]["defective"]["recall"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertEqual(result["roc_auc"], 0.75)
        self.assertEqual(result["positive_class"], "defective")
        self.assertEqual(result["confident_mistakes"][0]["image_key"], "image-2")

    def test_ties_and_pairwise_auc_reference(self) -> None:
        self.assertEqual(
            self.score([0, 0, 1, 1], [0.2, 0.5, 0.5, 0.8])["roc_auc"], 0.875
        )
        self.assertEqual(self.score([0, 1], [0.5, 0.5])["roc_auc"], 0.5)
        self.assertEqual(self.score([0, 1], [0.9, 0.1])["roc_auc"], 0.0)
        generator = np.random.default_rng(42)
        for _ in range(50):
            positive = np.array(
                [False, True] + generator.choice([False, True], size=20).tolist()
            )
            scores = generator.integers(0, 5, size=len(positive)) / 4
            comparisons = [
                float(p > n) + 0.5 * (p == n)
                for p in scores[positive]
                for n in scores[~positive]
            ]
            self.assertAlmostEqual(
                _binary_auc(positive, scores), float(np.mean(comparisons))
            )

    def test_missing_classes_are_explicit_and_json_has_no_nan(self) -> None:
        result = self.score([0, 0], [0.1, 0.2])
        self.assertIsNone(result["roc_auc"])
        self.assertEqual(result["per_class"]["defective"]["support"], 0)
        self.assertEqual(result["per_class"]["defective"]["f1"], 0)
        self.assertEqual(result["confident_mistakes"], [])
        json.dumps(result, allow_nan=False)

    def test_multiclass_one_vs_rest_and_macro_metrics(self) -> None:
        result = classification_metrics(
            np.array([0, 1, 2]),
            np.eye(3) * 0.7 + 0.1,
            ["a", "b", "c"],
            ("one", "two", "three"),
        )
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["roc_auc_method"], "macro_one_vs_rest")
        absent = classification_metrics(
            np.array([0, 1]),
            np.eye(3)[:2] * 0.7 + 0.1,
            ["a", "b"],
            ("one", "two", "three"),
        )
        self.assertIsNone(absent["roc_auc"])

    def test_mistakes_are_sorted_by_confidence_then_key_and_limited_to_16(self) -> None:
        keys = [f"image-{i:02d}" for i in reversed(range(20))]
        scores = np.tile([0.2, 0.8], (20, 1))
        scores[0] = [0.1, 0.9]
        result = classification_metrics(
            np.zeros(20, dtype=int), scores, keys, ("ok", "defective")
        )
        mistakes = result["confident_mistakes"]
        self.assertEqual(len(mistakes), 16)
        self.assertEqual(mistakes[0]["image_key"], "image-19")
        self.assertEqual(
            [row["image_key"] for row in mistakes[1:]], sorted(keys[1:])[:15]
        )

    def test_invalid_predictions_are_rejected(self) -> None:
        for targets, probabilities in (
            ([], []),
            ([2], [[0.5, 0.5]]),
            ([0.5], [[0.5, 0.5]]),
            ([0], [[float("nan"), 0.5]]),
            ([0], [[0.1, 0.1]]),
            ([0], [[-0.1, 1.1]]),
            ([0], [[1.0]]),
        ):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                classification_metrics(
                    np.array(targets),
                    np.array(probabilities),
                    ["a"],
                    ("ok", "defective"),
                )


if __name__ == "__main__":
    unittest.main()
