from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from prediction_fixture import PredictionAPITest
from sqlalchemy.orm import Session

from vf_api.models import Prediction, Project
from vf_api.upload_dependencies import get_predictions


class PredictionStatsTest(PredictionAPITest):
    def setUp(self):
        super().setUp()
        self.store = Mock()
        self.store.url.return_value = "https://storage.example/private-preview"
        self.app.dependency_overrides[get_predictions] = lambda: self.store
        self.addCleanup(self.app.dependency_overrides.clear)

    def stats(self, **params):
        return self.client.get(
            self.base + "/stats", headers=self.headers, params=params
        )

    def add(self, latency, confidence, *, output=None, hours=0):
        with Session(self.engine) as session:
            session.add(
                Prediction(
                    deployment_id=self.deployment_id,
                    image_key="private-image",
                    output=output
                    or {
                        "label": "red",
                        "confidence": confidence,
                        "scores": {"red": confidence, "green": 1 - confidence},
                    },
                    latency_ms=latency,
                    max_confidence=confidence,
                    sent_to_review=confidence < 0.6,
                    created_at=datetime.now(timezone.utc) - timedelta(hours=hours),
                )
            )
            session.commit()

    def test_empty_stats_are_zero_with_null_latency(self):
        response = self.stats()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["requests"], 0)
        self.assertIsNone(result["latency_p95_ms"])
        self.assertEqual(result["class_distribution"], {"red": 0, "green": 0})
        self.assertEqual(sum(b["count"] for b in result["confidence_histogram"]), 0)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_percentiles_histogram_window_and_paginated_history(self):
        for latency, confidence in ((10, 0), (20, 0.2), (30, 0.99), (40, 1)):
            self.add(latency, confidence)
        self.add(1000, 0.8, hours=25)
        result = self.stats().json()
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["low_confidence_count"], 2)
        self.assertEqual(result["latency_mean_ms"], 25)
        self.assertEqual(result["latency_p50_ms"], 25)
        self.assertAlmostEqual(result["latency_p95_ms"], 38.5)
        self.assertEqual(
            [b["count"] for b in result["confidence_histogram"]],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 2],
        )
        self.assertEqual(result["class_distribution"]["red"], 4)
        self.assertEqual(self.stats(window="7d").json()["requests"], 5)
        history = self.client.get(
            self.base + "/predictions?page_size=2&page=2", headers=self.headers
        )
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json()["total"], 5)
        self.assertEqual(len(history.json()["items"]), 2)
        self.assertNotIn("private-image", history.text)
        self.assertEqual(history.json()["items"][0]["image_url_expires_in"], 300)

    def test_detection_distribution_counts_boxes_and_empty_predictions(self):
        with Session(self.engine) as session:
            session.get(Project, self.project_id).task_type = "detection"
            session.commit()
        box = {
            "label": "green",
            "confidence": 0.8,
            "x": 0.1,
            "y": 0.1,
            "w": 0.2,
            "h": 0.2,
        }
        self.add(10, 0.8, output={"boxes": [box, box]})
        self.add(20, 0, output={"boxes": []})
        result = self.stats().json()
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["class_distribution"], {"red": 0, "green": 2})
        self.assertEqual(result["class_count_unit"], "boxes")
        self.assertEqual(result["low_confidence_count"], 1)

    def test_access_controls_and_query_validation(self):
        for suffix in ("/stats", "/predictions"):
            for headers, expected in (
                ({}, 401),
                (self.auth(self.user_id, "refresh"), 401),
                (self.auth(self.other_id), 404),
            ):
                self.assertEqual(
                    self.client.get(self.base + suffix, headers=headers).status_code,
                    expected,
                )
        self.store.url.assert_not_called()
        self.assertEqual(self.stats(window="forever").status_code, 422)
        self.assertEqual(
            self.client.get(
                self.base + "/predictions?page_size=101", headers=self.headers
            ).status_code,
            422,
        )
