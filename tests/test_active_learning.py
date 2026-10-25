"""Check reproducible acquisition and the recorded experiment's split boundary."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from scripts.active_learning import partition

ROOT = Path(__file__).resolve().parents[1]


class ActiveLearningTest(unittest.TestCase):
    def test_stratified_partition_is_order_independent_and_disjoint(self):
        rows = [
            {"image_key": f"{label}/{i:03}", "labels": [{"class_index": label}]}
            for label, count in ((0, 80), (1, 120))
            for i in range(count)
        ]
        initial, pool = partition(rows)
        self.assertEqual((initial, pool), partition(list(reversed(rows))))
        self.assertEqual(
            Counter(r["labels"][0]["class_index"] for r in initial), {0: 8, 1: 12}
        )

        def keys(data):
            return {r["image_key"] for r in data}

        self.assertFalse(keys(initial) & keys(pool))
        self.assertEqual(keys(initial) | keys(pool), keys(rows))
        self.assertNotEqual(initial, partition(rows, seed=43)[0])

    def test_recorded_acquisition_and_reports_are_consistent(self):
        result = json.loads(
            (ROOT / "docs/results/active-learning-20261025.json").read_text()
        )
        path = ROOT / "docs/results/active-learning-acquisition-20261025.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), result["acquisition_sha256"]
        )
        acquisition = json.loads(path.read_text())
        ranked = acquisition["ranked_pool"]
        self.assertEqual(
            ranked, sorted(ranked, key=lambda r: (r["confidence"], r["image_key"]))
        )
        keys = {r["image_key"] for r in ranked}
        self.assertEqual(len(keys), result["protocol"]["pool_images"])
        self.assertTrue(all(0 <= r["confidence"] <= 1 for r in ranked))
        uncertain, random = map(
            set, (acquisition["uncertainty_keys"], acquisition["random_keys"])
        )
        self.assertEqual(uncertain, {r["image_key"] for r in ranked[:200]})
        self.assertEqual(len(random), 200)
        self.assertTrue(random <= keys)
        self.assertEqual(len(uncertain & random), result["acquisition_overlap"])
        for name, arm in result["arms"].items():
            self.assertEqual(arm["training_images"], 558 if name == "seed" else 758)
            self.assertEqual(arm["metrics"]["samples"], 715)
            self.assertEqual(
                sum(map(sum, arm["metrics"]["confusion_matrix"]["counts"])), 715
            )
            self.assertEqual(arm["run"]["config"], result["protocol"]["config"])


if __name__ == "__main__":
    unittest.main()
