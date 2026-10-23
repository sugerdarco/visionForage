"""Authenticated and bounded HTTP inference with a tiny real ONNX graph."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from bundle_fixture import make_bundle
from test_runtime import picture
from vf_inference.main import create_app
from vf_inference.settings import Settings


class InferenceHTTPTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        hashes = make_bundle(temporary.name)
        self.key = "test-only-inference-key-" + uuid4().hex
        self.settings = Settings(
            bundle_dir=Path(temporary.name),
            model_id=uuid4(),
            model_version=3,
            slug="test-model",
            model_sha256=hashes[0],
            metadata_sha256=hashes[1],
            api_key_sha256=hashlib.sha256(self.key.encode()).hexdigest(),
        )
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"X-API-Key": self.key}

    def post(self, **kwargs):
        return self.client.post("/predict/test-model", headers=self.headers, **kwargs)

    def test_predict_health_and_openapi_contract(self):
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        response = self.post(files={"image": ("image.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["model_id"], str(self.settings.model_id))
        self.assertEqual(data["model_version"], 3)
        self.assertEqual(data["label"], "red")
        self.assertIn("request_id", data)
        self.assertNotIn("prediction_id", data)
        self.assertGreater(data["latency_ms"], 0)
        self.assertEqual(response.headers["cache-control"], "no-store")
        spec = self.client.get("/openapi.json").json()
        self.assertEqual(
            spec["paths"]["/predict/{slug}"]["post"]["security"], [{"APIKeyHeader": []}]
        )

    def test_auth_and_slug_checks_precede_parsing(self):
        for headers in ({}, {"X-API-Key": "incorrect"}, {"X-API-Key": "x" * 513}):
            self.assertEqual(
                self.client.post(
                    "/predict/test-model", headers=headers, content=b"invalid"
                ).status_code,
                401,
            )
        self.assertEqual(
            self.client.post(
                "/predict/other", headers=self.headers, content=b"invalid"
            ).status_code,
            404,
        )

    def test_rejects_invalid_and_oversized_uploads_and_releases_slot(self):
        for options, status in (
            ({"content": b"wrong"}, 415),
            ({"files": {"wrong": ("i.png", picture(), "image/png")}}, 422),
            ({"files": {"image": ("i.txt", b"text", "text/plain")}}, 415),
            ({"files": {"image": ("i.png", b"invalid", "image/png")}}, 422),
            ({"files": {"image": ("i.jpg", picture(), "image/jpeg")}}, 422),
            (
                {
                    "files": [
                        ("image", ("a.png", picture(), "image/png")),
                        ("image", ("b.png", picture(), "image/png")),
                    ]
                },
                422,
            ),
        ):
            with self.subTest(status=status):
                self.assertEqual(self.post(**options).status_code, status)
        with patch("vf_inference.main.MAX_BODY_BYTES", 64):
            self.assertEqual(
                self.post(
                    files={"image": ("i.png", picture(), "image/png")}
                ).status_code,
                413,
            )
        with patch("vf_inference.main.MAX_IMAGE_BYTES", 1):
            self.assertEqual(
                self.post(
                    files={"image": ("i.png", picture(), "image/png")}
                ).status_code,
                413,
            )
        self.assertEqual(
            self.post(files={"image": ("i.png", picture(), "image/png")}).status_code,
            200,
        )

    def test_busy_request_is_rejected_and_capacity_recovers(self):
        started, release = Event(), Event()
        predict = self.app.state.runtime.predict

        def slow(*args):
            started.set()
            release.wait(10)
            return predict(*args)

        with patch.object(self.app.state.runtime, "predict", side_effect=slow):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self.post, files={"image": ("i.png", picture(), "image/png")}
                )
                try:
                    self.assertTrue(started.wait(5))
                    busy = self.post(content=b"not even parsed")
                    self.assertEqual(busy.status_code, 429)
                    self.assertEqual(busy.headers["retry-after"], "1")
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=10).status_code, 200)
        self.assertEqual(
            self.post(files={"image": ("i.png", picture(), "image/png")}).status_code,
            200,
        )

    def test_runtime_errors_do_not_expose_bundle_paths(self):
        with patch.object(
            self.app.state.runtime,
            "predict",
            side_effect=RuntimeError("private runtime detail"),
        ):
            response = self.post(files={"image": ("i.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private runtime detail", response.text)

    def test_invalid_bundle_prevents_startup(self):
        app = create_app(self.settings.model_copy(update={"model_sha256": "0" * 64}))
        with self.assertRaisesRegex(RuntimeError, "Cannot initialize"):
            with TestClient(app):
                pass

    def test_invalid_key_configuration_does_not_echo_secret_input(self):
        from pydantic import ValidationError

        values = self.settings.model_dump()
        values["api_key_sha256"] = "private-plaintext-mistake"
        with self.assertRaises(ValidationError) as caught:
            Settings(**values)
        self.assertNotIn("private-plaintext-mistake", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
