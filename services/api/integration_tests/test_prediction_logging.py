"""Real PostgreSQL/MinIO retries and delivery by the existing CPU worker."""

from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from botocore.exceptions import EndpointConnectionError
from celery.contrib.testing.worker import start_worker
import httpx2 as httpx
from PIL import Image
from prediction_fixture import PredictionAPITest
from redis import Redis
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from vf_api.database import make_session_factory
from vf_api.models import Deployment, Prediction, Project
from vf_api.queueing import PREDICTION_TASK
from vf_api.settings import DeploymentSettings, QueueSettings, StorageSettings
from vf_api.storage import PredictionStore
from vf_api.upload_dependencies import get_predictions
from vf_worker.app import create_worker
from vf_worker.predictions import collect_predictions


class PredictionLoggingTest(PredictionAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.storage = StorageSettings(
            s3_predictions_bucket="vf-test-pred-" + uuid4().hex
        )
        cls.store = PredictionStore(cls.storage)
        cls.addClassCleanup(cls.store.close)
        cls.store.client.create_bucket(Bucket=cls.store.bucket)
        cls.addClassCleanup(cls.drop_bucket)

    @classmethod
    def drop_bucket(cls):
        for page in cls.store.client.get_paginator("list_objects_v2").paginate(
            Bucket=cls.store.bucket
        ):
            objects = [{"Key": row["Key"]} for row in page.get("Contents", [])]
            if objects:
                cls.store.client.delete_objects(
                    Bucket=cls.store.bucket, Delete={"Objects": objects}
                )
        cls.store.client.delete_bucket(Bucket=cls.store.bucket)

    def setUp(self):
        super().setUp()
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / str(self.deployment_id)
        self.directory.mkdir()
        self.sessions = make_session_factory(self.engine)
        output = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(output, format="PNG")
        self.image = output.getvalue()
        self.app.dependency_overrides[get_predictions] = lambda: self.store
        self.addCleanup(self.app.dependency_overrides.clear)

    def event(self, output=None):
        identifier = uuid4()
        data = {
            "schema_version": 1,
            "prediction_id": str(identifier),
            "deployment_id": str(self.deployment_id),
            "model_id": str(self.model_id),
            "model_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mime": "image/png",
            "image_bytes": len(self.image),
            "image_sha256": hashlib.sha256(self.image).hexdigest(),
            "latency_ms": 12.5,
            "output": output
            or {
                "label": "red",
                "confidence": 0.55,
                "scores": {"red": 0.55, "green": 0.45},
            },
        }
        path = self.directory / f"{identifier}.json"
        path.write_text(json.dumps(data))
        path.with_suffix(".image").write_bytes(self.image)
        return identifier, path

    def collect(self):
        return collect_predictions(self.root, self.sessions, self.store)

    def count(self):
        with self.sessions() as session:
            return session.scalar(
                select(func.count())
                .select_from(Prediction)
                .where(Prediction.deployment_id == self.deployment_id)
            )

    def test_persistence_preview_and_replay_after_commit(self):
        identifier, path = self.event()
        original = path.read_bytes()
        self.assertEqual(self.collect()["collected"], 1)
        with self.sessions() as session:
            row = session.get(Prediction, identifier)
            self.assertTrue(row.sent_to_review)
            self.assertEqual(row.latency_ms, 12.5)
            self.assertEqual(self.store.read(row.image_key), self.image)
        # Crash after removing the image but before removing the committed event.
        path.write_bytes(original)
        self.assertEqual(self.collect()["collected"], 1)
        self.assertEqual(self.count(), 1)
        self.assertFalse(path.exists())
        response = self.client.get(self.base + "/predictions", headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["id"], str(identifier))
        with httpx.Client(trust_env=False) as client:
            self.assertEqual(client.get(item["image_url"]).content, self.image)
            split = urlsplit(item["image_url"])
            self.assertEqual(
                client.get(
                    urlunsplit((split.scheme, split.netloc, split.path, "", ""))
                ).status_code,
                403,
            )
        self.assertEqual(
            self.client.get(self.base + "/stats", headers=self.headers).json()[
                "requests"
            ],
            1,
        )

    def test_storage_and_database_outages_keep_event_for_retry(self):
        identifier, path = self.event()
        with patch.object(
            self.store,
            "put_once",
            side_effect=EndpointConnectionError(endpoint_url="http://unavailable"),
        ):
            with self.assertRaises(EndpointConnectionError):
                self.collect()
        self.assertTrue(path.exists())
        self.assertEqual(self.count(), 0)
        with patch.object(
            Session,
            "commit",
            side_effect=OperationalError("commit", {}, Exception("unavailable")),
        ):
            with self.assertRaises(OperationalError):
                self.collect()
        self.assertTrue(path.exists())
        self.assertEqual(self.count(), 0)
        # The image already exists in MinIO; retry accepts identical bytes only.
        self.assertEqual(self.collect()["collected"], 1)
        self.assertEqual(self.count(), 1)

    def test_invalid_identity_checksum_and_nonfinite_values_are_retained(self):
        _, path = self.event()
        original = json.loads(path.read_text())
        for field, value in (
            ("model_id", str(uuid4())),
            ("deployment_id", str(uuid4())),
            ("latency_ms", float("nan")),
            ("image_sha256", "0" * 64),
        ):
            path.write_text(json.dumps({**original, field: value}))
            self.assertEqual(self.collect(), {"collected": 0, "invalid": 1})
            self.assertTrue(path.exists())
        self.assertEqual(self.count(), 0)
        path.write_text(json.dumps(original))
        self.assertEqual(self.collect()["collected"], 1)

    def test_detection_empty_results_and_threshold_boundary_after_stop(self):
        with self.sessions() as session:
            session.get(Project, self.project_id).task_type = "detection"
            session.get(Deployment, self.deployment_id).status = "stopped"
            session.commit()
        empty, _ = self.event({"boxes": []})
        boundary, _ = self.event(
            {
                "boxes": [
                    {
                        "label": "green",
                        "confidence": 0.6,
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.2,
                        "h": 0.2,
                    }
                ]
            }
        )
        self.assertEqual(self.collect()["collected"], 2)
        with self.sessions() as session:
            self.assertTrue(session.get(Prediction, empty).sent_to_review)
            self.assertFalse(session.get(Prediction, boundary).sent_to_review)

    def test_collector_lock_and_partial_writes(self):
        _, path = self.event()
        with (self.root / ".collector.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.collect()["collected"], 0)
        temporary = path.with_suffix(".tmp")
        path.rename(temporary)
        self.assertEqual(self.collect()["collected"], 0)
        temporary.rename(path)
        self.assertEqual(self.collect()["collected"], 1)

    def test_real_cpu_queue_delivers_collection_task(self):
        self.event()
        queue = QueueSettings()
        worker = create_worker(
            database=self.settings,
            storage=self.storage,
            queue=queue,
            deployments=DeploymentSettings(prediction_log_dir=self.root),
        )
        self.addCleanup(worker.close)
        prefix = "vf-test-pred-" + uuid4().hex + ":"
        worker.conf.task_default_queue = prefix + "cpu"
        worker.conf.broker_transport_options = {
            **worker.conf.broker_transport_options,
            "global_keyprefix": prefix,
        }

        def cleanup():
            with Redis.from_url(queue.broker_url) as connection:
                keys = list(connection.scan_iter(match=prefix + "*"))
                if keys:
                    connection.delete(*keys)

        self.addCleanup(cleanup)
        self.assertEqual(
            worker.conf.beat_schedule["collect-predictions"]["task"], PREDICTION_TASK
        )
        with start_worker(
            worker,
            perform_ping_check=False,
            pool="solo",
            queues=[worker.conf.task_default_queue],
        ):
            worker.send_task(PREDICTION_TASK)
            deadline = time.monotonic() + 10
            while self.count() != 1:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)
