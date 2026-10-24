"""Review acceptance and project cleanup against real PostgreSQL and MinIO."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

from botocore.exceptions import EndpointConnectionError
from PIL import Image as PILImage
from prediction_fixture import PredictionAPITest
from sqlalchemy import func, select

from vf_api.database import make_session_factory
from vf_api.deployment_manager import DeploymentManager
from vf_api.deployment_runtime import DeploymentUnavailable
from vf_api.models import (
    Annotation,
    Class,
    DatasetVersion,
    Image,
    Model,
    Prediction,
    Project,
    TrainingJob,
)
from vf_api.routers.deployments import operation_lock
from vf_api.settings import DeploymentSettings, StorageSettings
from vf_api.storage import ArtifactStore, DatasetStore, ObjectStore, PredictionStore
from vf_api.upload_dependencies import (
    get_datasets,
    get_predictions,
    get_queue,
    get_store,
)
from vf_worker.project_cleanup import clean_project


class ReviewCleanupTest(PredictionAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        suffix = uuid4().hex
        settings = StorageSettings(
            s3_images_bucket="vf-test-review-images-" + suffix,
            s3_datasets_bucket="vf-test-review-data-" + suffix,
            s3_artifacts_bucket="vf-test-review-models-" + suffix,
            s3_predictions_bucket="vf-test-review-pred-" + suffix,
        )
        cls.stores = {
            "images": ObjectStore(settings),
            "datasets": DatasetStore(settings),
            "artifacts": ArtifactStore(settings),
            "predictions": PredictionStore(settings),
        }
        for store in cls.stores.values():
            cls.addClassCleanup(store.close)
            store.client.create_bucket(Bucket=store.bucket)
            cls.addClassCleanup(cls.drop_bucket, store)

    @classmethod
    def drop_bucket(cls, store):
        for page in store.client.get_paginator("list_objects_v2").paginate(
            Bucket=store.bucket
        ):
            objects = [{"Key": row["Key"]} for row in page.get("Contents", [])]
            if objects:
                store.client.delete_objects(
                    Bucket=store.bucket, Delete={"Objects": objects}
                )
        store.client.delete_bucket(Bucket=store.bucket)

    def setUp(self):
        super().setUp()
        self.sessions = make_session_factory(self.engine)
        self.queue = Mock()
        self.app.dependency_overrides.update(
            {
                get_store: lambda: self.stores["images"],
                get_datasets: lambda: self.stores["datasets"],
                get_predictions: lambda: self.stores["predictions"],
                get_queue: lambda: self.queue,
            }
        )
        self.addCleanup(self.app.dependency_overrides.clear)
        with self.sessions() as session:
            rows = [
                Class(
                    project_id=self.project_id, name=name, index=index, color="#112233"
                )
                for index, name in enumerate(("red", "green"))
            ]
            session.add_all(rows)
            session.flush()
            self.labels = [row.id for row in rows]
            session.commit()
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = Mock()
        self.manager = DeploymentManager(
            DeploymentSettings(
                deployment_work_dir=self.root / "bundles",
                prediction_log_dir=self.root / "predictions",
            ),
            self.runtime,
        )

    def prediction(self, color=(200, 0, 0), *, eligible=True):
        buffer = io.BytesIO()
        PILImage.new("RGB", (32, 32), color).save(buffer, format="PNG")
        data = buffer.getvalue()
        identifier = uuid4()
        key = f"{self.deployment_id}/2026-10-23/{identifier}.png"
        self.stores["predictions"].put_once(
            key, data, "image/png", hashlib.sha256(data).hexdigest()
        )
        with self.sessions() as session:
            session.add(
                Prediction(
                    id=identifier,
                    deployment_id=self.deployment_id,
                    image_key=key,
                    output={
                        "label": "red",
                        "confidence": 0.55,
                        "scores": {"red": 0.55, "green": 0.45},
                    },
                    max_confidence=0.55,
                    latency_ms=12,
                    sent_to_review=eligible,
                )
            )
            session.commit()
        return identifier

    def accept(self, identifier, label=0, *, annotations=None, headers=None):
        return self.client.post(
            f"/api/v1/review/{identifier}/accept",
            headers=self.headers if headers is None else headers,
            json={
                "annotations": annotations
                if annotations is not None
                else [{"class_id": str(self.labels[label])}]
            },
        )

    def cleanup(self):
        return clean_project(self.project_id, self.sessions, self.stores, self.manager)

    def test_review_accepts_once_and_next_version_includes_corrected_images(self):
        identifiers = []
        for label in (0, 1):
            for index in range(3):
                identifier = self.prediction((100 + index, label * 150, 10))
                identifiers.append(identifier)
                accepted = self.accept(identifier, label)
                self.assertEqual(accepted.status_code, 200, accepted.text)
                self.assertEqual(self.accept(identifier, label).json(), accepted.json())
        pending = self.client.get(
            f"/api/v1/projects/{self.project_id}/review-queue", headers=self.headers
        )
        self.assertEqual(pending.json()["total"], 0)
        with self.sessions() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count())
                    .select_from(Image)
                    .where(Image.project_id == self.project_id)
                ),
                6,
            )
            self.assertEqual(
                session.get(DatasetVersion, self.version_id).counts, self.counts
            )
        version = self.client.post(
            f"/api/v1/projects/{self.project_id}/dataset-versions",
            headers=self.headers,
            json={},
        )
        self.assertEqual(version.status_code, 201, version.text)
        self.assertEqual(version.json()["version"], 2)
        self.assertEqual(sum(version.json()["counts"]["images"].values()), 6)
        self.assertEqual(self.accept(identifiers[0], 1).status_code, 409)
        self.assertEqual(
            self.client.get(self.base + "/stats", headers=self.headers).json()[
                "low_confidence_count"
            ],
            6,
        )

    def test_concurrent_acceptance_and_image_deduplication_conflicts(self):
        first, second = self.prediction(), self.prediction()
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(self.accept, [first, first]))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self.accept(second, 1).status_code, 409)
        self.assertEqual(
            self.accept(second).json()["image_id"], responses[0].json()["image_id"]
        )

    def test_ownership_label_validation_and_storage_failure_leave_review_pending(self):
        identifier = self.prediction()
        for headers, expected in (({}, 401), (self.auth(self.other_id), 404)):
            self.assertEqual(
                self.accept(identifier, headers=headers).status_code, expected
            )
        self.assertEqual(self.accept(identifier, annotations=[]).status_code, 422)
        self.assertEqual(
            self.accept(
                identifier, annotations=[{"class_id": str(uuid4())}]
            ).status_code,
            422,
        )
        with patch.object(
            self.stores["images"],
            "put",
            side_effect=EndpointConnectionError(endpoint_url="http://unavailable"),
        ):
            self.assertEqual(self.accept(identifier).status_code, 503)
        with self.sessions() as session:
            self.assertIsNone(session.get(Prediction, identifier).reviewed_at)
        self.assertEqual(self.accept(self.prediction(eligible=False)).status_code, 409)
        self.assertEqual(
            self.client.get(
                f"/api/v1/projects/{self.project_id}/review-queue",
                headers=self.auth(self.other_id),
            ).status_code,
            404,
        )

    def test_reviewed_negative_detection_and_human_box_labels(self):
        with self.sessions() as session:
            session.get(Project, self.project_id).task_type = "detection"
            session.commit()
        first = self.accept(self.prediction(), annotations=[])
        self.assertEqual(first.status_code, 200, first.text)
        second = self.accept(
            self.prediction((1, 2, 3)),
            annotations=[
                {"class_id": str(self.labels[1]), "bbox": [0.1, 0.2, 0.3, 0.4]}
            ],
        )
        self.assertEqual(second.status_code, 200, second.text)
        with self.sessions() as session:
            self.assertEqual(
                session.get(Image, UUID(first.json()["image_id"])).status, "labeled"
            )
            annotation = session.scalar(
                select(Annotation).where(
                    Annotation.image_id == UUID(second.json()["image_id"])
                )
            )
            self.assertEqual(annotation.source, "human")
            self.assertIsNone(annotation.confidence)

    def test_populated_deletion_removes_only_owned_namespaces_and_rows(self):
        accepted = self.accept(self.prediction())
        self.assertEqual(accepted.status_code, 200)
        for name in ("images", "datasets", "artifacts"):
            self.stores[name].put(f"{self.project_id}/orphan", b"owned", "text/plain")
            self.stores[name].put(
                f"{self.other_project_id}/keep", b"keep", "text/plain"
            )
        for prefix in (
            f"uploads/{self.project_id}/",
            f"uploads/imports/{self.project_id}/",
        ):
            self.stores["images"].put(prefix + "staged", b"staged", "text/plain")
        outbox = self.manager.settings.prediction_log_dir / str(self.deployment_id)
        outbox.mkdir(parents=True, mode=0o700)
        self.manager.settings.prediction_log_dir.chmod(0o700)
        (outbox / "partial.image").write_bytes(b"pending")
        requested = self.client.delete(
            f"/api/v1/projects/{self.project_id}", headers=self.headers
        )
        self.assertEqual(requested.status_code, 202, requested.text)
        self.assertEqual(self.accept(self.prediction((2, 3, 4))).status_code, 409)
        self.assertTrue(self.cleanup())
        self.assertFalse(outbox.exists())
        self.assertEqual(
            self.client.get(
                f"/api/v1/projects/{self.project_id}", headers=self.headers
            ).status_code,
            404,
        )
        with self.sessions() as session:
            self.assertIsNone(session.get(Model, self.model_id))
            self.assertIsNotNone(session.get(Project, self.other_project_id))
        for name in ("images", "datasets", "artifacts"):
            self.assertEqual(
                self.stores[name].read(f"{self.other_project_id}/keep"), b"keep"
            )
            self.assertFalse(
                self.stores[name]
                .client.list_objects_v2(
                    Bucket=self.stores[name].bucket, Prefix=f"{self.project_id}/"
                )
                .get("Contents")
            )
        self.assertFalse(self.cleanup())

    def test_deletion_cancels_jobs_and_waits_for_worker_and_deployment_operations(self):
        with self.sessions() as session:
            job = TrainingJob(
                project_id=self.project_id,
                dataset_version_id=self.version_id,
                model_arch="resnet18",
                status="running",
            )
            queued = TrainingJob(
                project_id=self.project_id,
                dataset_version_id=self.version_id,
                model_arch="resnet18",
                status="queued",
            )
            session.add_all([job, queued])
            session.flush()
            running_id, queued_id = job.id, queued.id
            session.commit()
        path = f"/api/v1/projects/{self.project_id}"
        self.assertEqual(
            self.client.delete(path, headers=self.auth(self.other_id)).status_code, 404
        )
        self.assertEqual(
            self.client.delete(path, headers=self.headers).status_code, 202
        )
        self.assertFalse(self.cleanup())
        self.runtime.remove.assert_not_called()
        with self.sessions() as session:
            self.assertEqual(session.get(TrainingJob, queued_id).status, "cancelled")
            self.assertIsNotNone(
                session.get(TrainingJob, running_id).cancel_requested_at
            )
            session.get(TrainingJob, running_id).status = "cancelled"
            session.commit()
        with self.sessions() as session:
            self.assertTrue(operation_lock(session, self.deployment_id))
            self.assertFalse(self.cleanup())
        with patch.object(
            self.runtime, "remove", side_effect=DeploymentUnavailable("private")
        ):
            with self.assertRaises(DeploymentUnavailable):
                self.cleanup()
        with patch.object(
            self.stores["datasets"],
            "delete_prefix",
            side_effect=EndpointConnectionError(endpoint_url="http://unavailable"),
        ):
            with self.assertRaises(EndpointConnectionError):
                self.cleanup()
        self.assertTrue(self.client.get(path, headers=self.headers).json()["deleting"])
        self.assertTrue(self.cleanup())
