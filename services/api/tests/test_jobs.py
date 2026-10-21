"""Training request validation, isolation, metrics and cancellation on PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from uuid import UUID, uuid4

from job_fixture import JobAPITest
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.models import Project, TrainingJob, TrainingMetric


class JobTest(JobAPITest):
    def test_create_freezes_effective_defaults_without_external_services(self):
        response = self.create(hyperparams={"epochs": 3})
        self.assertEqual(response.status_code, 201, response.text)
        row = response.json()
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["dataset_version_id"], str(self.version_id))
        self.assertEqual(
            row["hyperparams"],
            {
                "seed": 42,
                "epochs": 3,
                "batch_size": 64,
                "patience": 5,
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "label_smoothing": 0.1,
                "horizontal_flip": False,
                "vertical_flip": False,
            },
        )
        for name in (
            "started_at",
            "finished_at",
            "dispatched_at",
            "cancel_requested_at",
            "gpu_id",
            "error",
        ):
            self.assertIsNone(row[name])
        self.assertNotIn("celery_task_id", row)
        with Session(self.engine) as session:
            stored = session.get(TrainingJob, UUID(row["id"]))
            self.assertIsNotNone(UUID(stored.celery_task_id))
        self.assertEqual(self.read(row["id"]).json(), row)
        self.assertEqual(self.create().status_code, 201)
        listing = self.client.get(
            self.path + "?page_size=1", headers=self.headers
        ).json()
        self.assertEqual(listing["total"], 2)
        second = self.client.get(
            self.path + "?page=2&page_size=1", headers=self.headers
        ).json()
        self.assertEqual(second["items"][0]["id"], row["id"])

    def test_detector_defaults_and_model_task_validation(self):
        self.assertEqual(self.create(model_arch="yolov8s").status_code, 422)
        with Session(self.engine) as session:
            session.get(Project, self.project_id).task_type = "detection"
            session.commit()
        self.assertEqual(self.create().status_code, 422)
        response = self.create(model_arch="yolov8s")
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(
            response.json()["hyperparams"],
            {
                "seed": 42,
                "epochs": 100,
                "batch_size": 16,
                "patience": 20,
                "image_size": 640,
                "horizontal_flip": False,
                "vertical_flip": False,
            },
        )
        for value in (63, 65, 1312, True, "640"):
            self.assertEqual(
                self.create(
                    model_arch="yolov8s", hyperparams={"image_size": value}
                ).status_code,
                422,
            )

    def test_hyperparameters_are_bounded_and_cannot_control_worker_environment(self):
        for params in (
            {"epochs": True},
            {"epochs": "5"},
            {"epochs": 0},
            {"epochs": 1001},
            {"batch_size": 257},
            {"batch_size": -1},
            {"patience": 0},
            {"seed": -1},
            {"seed": 2**32},
            {"horizontal_flip": 1},
            {"learning_rate": 0},
            {"learning_rate": True},
            {"weight_decay": -1},
            {"label_smoothing": 1},
            {"image_size": 640},
            {"device": "cuda"},
            {"workers": 99},
            {"output": "/tmp/path"},
            {"pretrained": False},
            {"fast_dev_run": True},
            {"command": "ignored"},
        ):
            with self.subTest(params=params):
                self.assertEqual(self.create(hyperparams=params).status_code, 422)
        for overrides in (
            {"model_arch": "resnet50"},
            {"gpu_id": 1},
            {"status": "succeeded"},
        ):
            self.assertEqual(self.create(**overrides).status_code, 422)
        response = self.client.post(
            self.path,
            headers=self.headers,
            content=json.dumps(
                {
                    "dataset_version_id": str(self.version_id),
                    "model_arch": "resnet18",
                    "hyperparams": {"learning_rate": float("nan")},
                }
            ),
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self.client.get(self.path, headers=self.headers).json()["total"], 0
        )

    def test_rejects_old_unverified_versions_and_insufficient_test_images(self):
        variants = [
            {},
            {**self.counts, "audit_sha256": "bad"},
            {**self.counts, "images": {"train": 2, "val": 1, "test": 31}},
            {**self.counts, "images": {"train": 0, "val": 1, "test": 32}},
            {**self.counts, "images": {"train": 2, "val": 0, "test": 32}},
        ]
        for number, counts in enumerate(variants, 2):
            with Session(self.engine) as session:
                version = self.add_version(
                    session, self.project_id, number=number, counts=counts
                )
                identifier = version.id
                session.commit()
            self.assertEqual(
                self.create(dataset_version_id=str(identifier)).status_code, 422
            )
        self.assertEqual(self.create(dataset_version_id=str(uuid4())).status_code, 404)
        self.assertEqual(
            self.create(dataset_version_id=str(self.other_version_id)).status_code, 404
        )

    def test_every_endpoint_checks_owner_and_access_token(self):
        identifier = self.create().json()["id"]
        routes = [
            (
                "POST",
                self.path,
                {"dataset_version_id": str(self.version_id), "model_arch": "resnet18"},
            ),
            ("GET", self.path, None),
            ("GET", f"/api/v1/jobs/{identifier}", None),
            ("GET", f"/api/v1/jobs/{identifier}/metrics", None),
            ("POST", f"/api/v1/jobs/{identifier}/cancel", None),
        ]
        for method, path, body in routes:
            for headers in ({}, self.auth(self.user_id, "refresh")):
                response = self.client.request(method, path, headers=headers, json=body)
                self.assertEqual(response.status_code, 401, response.text)
            response = self.client.request(
                method, path, headers=self.auth(self.other_id), json=body
            )
            self.assertEqual(response.status_code, 404, response.text)
        for suffix, method in (("", "GET"), ("/metrics", "GET"), ("/cancel", "POST")):
            response = self.client.request(
                method, f"/api/v1/jobs/{uuid4()}{suffix}", headers=self.headers
            )
            self.assertEqual(response.status_code, 404)
        self.assertEqual(self.read(identifier).json()["status"], "queued")

    def test_queued_cancellation_is_terminal_and_idempotent(self):
        identifier = self.create().json()["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: self.cancel(identifier), range(2)))
        self.assertEqual([row.status_code for row in responses], [200, 200])
        first, second = [row.json() for row in responses]
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "cancelled")
        self.assertIsNone(first["started_at"])
        self.assertEqual(first["finished_at"], first["cancel_requested_at"])

    def test_running_cancellation_waits_for_worker_and_terminal_jobs_stay_unchanged(
        self,
    ):
        for status in ("running", "succeeded", "failed"):
            identifier = self.create().json()["id"]
            with Session(self.engine) as session:
                row = session.get(TrainingJob, UUID(identifier))
                row.status = status
                row.started_at = datetime.now(timezone.utc)
                if status != "running":
                    row.finished_at = row.started_at
                session.commit()
            before = self.read(identifier).json()
            after = self.cancel(identifier).json()
            self.assertEqual(after["status"], status)
            if status == "running":
                self.assertIsNotNone(after["cancel_requested_at"])
                self.assertIsNone(after["finished_at"])
                self.assertEqual(after, self.cancel(identifier).json())
            else:
                self.assertEqual(before, after)

    def test_metrics_are_paginated_and_ordered_by_epoch_split_name(self):
        identifier = self.create().json()["id"]
        path = f"/api/v1/jobs/{identifier}/metrics"
        self.assertEqual(
            self.client.get(path, headers=self.headers).json()["items"], []
        )
        with Session(self.engine) as session:
            session.add_all(
                [
                    TrainingMetric(
                        job_id=UUID(identifier),
                        epoch=epoch,
                        split=split,
                        name=name,
                        value=0.5,
                    )
                    for epoch, split, name in (
                        (2, "val", "loss"),
                        (1, "val", "loss"),
                        (1, "train", "loss"),
                        (1, "train", "accuracy"),
                    )
                ]
            )
            session.commit()
        page = self.client.get(path + "?page_size=2", headers=self.headers).json()
        self.assertEqual(page["total"], 4)
        self.assertEqual([row["name"] for row in page["items"]], ["accuracy", "loss"])
        page2 = self.client.get(
            path + "?page_size=2&page=2", headers=self.headers
        ).json()
        self.assertEqual([row["epoch"] for row in page2["items"]], [1, 2])
        for route in (path, self.path):
            for query in ("page=0", "page_size=101", "page=bad"):
                self.assertEqual(
                    self.client.get(
                        route + "?" + query, headers=self.headers
                    ).status_code,
                    422,
                )
        with Session(self.engine) as session:
            self.assertEqual(
                len(
                    list(
                        session.scalars(
                            select(TrainingMetric).where(
                                TrainingMetric.job_id == UUID(identifier)
                            )
                        )
                    )
                ),
                4,
            )

    def test_openapi_has_typed_training_inputs_and_security(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        for path, methods in (
            ("/api/v1/projects/{project_id}/jobs", ("get", "post")),
            ("/api/v1/jobs/{job_id}", ("get",)),
            ("/api/v1/jobs/{job_id}/metrics", ("get",)),
            ("/api/v1/jobs/{job_id}/cancel", ("post",)),
        ):
            for method in methods:
                self.assertEqual(paths[path][method]["security"], [{"HTTPBearer": []}])
