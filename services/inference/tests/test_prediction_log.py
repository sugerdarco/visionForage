import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import unittest
import test_http
from test_runtime import picture


class PredictionLogTest(unittest.TestCase):
    post = test_http.InferenceHTTPTest.post

    def setUp(self):
        test_http.InferenceHTTPTest.setUp(self)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.outbox = Path(directory.name)
        self.settings.prediction_log_dir = self.outbox
        self.settings.deployment_id = uuid4()

    def test_record_is_complete_before_success_and_uses_response_identity(self):
        response = self.post(files={"image": ("image.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        identifier = response.json()["prediction_id"]
        event = json.loads((self.outbox / f"{identifier}.json").read_text())
        self.assertEqual(event["prediction_id"], response.json()["request_id"])
        self.assertEqual(event["deployment_id"], str(self.settings.deployment_id))
        self.assertEqual((self.outbox / f"{identifier}.image").read_bytes(), picture())
        self.assertEqual(event["output"]["label"], response.json()["label"])
        self.assertNotIn(self.key, json.dumps(event))

    def test_log_write_failure_and_full_outbox_return_503(self):
        with patch(
            "vf_inference.prediction_log.os.fsync", side_effect=OSError("private path")
        ):
            response = self.post(files={"image": ("image.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private path", response.text)
        self.assertFalse(list(self.outbox.iterdir()))
        with patch("vf_inference.prediction_log.MAX_PENDING", 0):
            response = self.post(files={"image": ("image.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(list(self.outbox.iterdir()))
