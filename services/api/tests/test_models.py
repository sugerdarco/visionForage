"""Registry reads against PostgreSQL, with frozen metadata and owner isolation."""

from uuid import uuid4

from job_fixture import JobAPITest
from sqlalchemy.orm import Session

from vf_api.models import Model, Project, TrainingJob


class ModelTest(JobAPITest):
    def add_model(self, *, version=1, other=False, promoted=False):
        project = self.other_project_id if other else self.project_id
        dataset = self.other_version_id if other else self.version_id
        with Session(self.engine) as session:
            job = TrainingJob(
                project_id=project,
                dataset_version_id=dataset,
                model_arch="yolov8s" if other else "resnet18",
                hyperparams={"epochs": 5, "seed": 42},
                status="succeeded",
            )
            session.add(job)
            session.flush()
            row = Model(
                project_id=project,
                job_id=job.id,
                version=version,
                input_size=640 if other else 224,
                class_map={"ok": 0, "defect": 1},
                metrics={"artifacts": {"private": "do-not-expose-storage-records"}},
                weights_key="private-weights-key",
                onnx_key="private-onnx-key",
                is_promoted=promoted,
            )
            session.add(row)
            session.flush()
            identifier, job_id = row.id, job.id
            session.commit()
        return identifier, job_id

    def listing(self, **params):
        return self.client.get(
            f"/api/v1/projects/{self.project_id}/models",
            headers=self.headers,
            params=params,
        )

    def detail(self, identifier, headers=None):
        return self.client.get(
            f"/api/v1/models/{identifier}",
            headers=self.headers if headers is None else headers,
        )

    def test_empty_registry_does_not_treat_jobs_as_models(self):
        self.assertEqual(self.create().status_code, 201)
        response = self.listing()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(), {"items": [], "total": 0, "page": 1, "page_size": 20}
        )

    def test_pagination_orders_versions_and_is_scoped_to_project(self):
        first, _ = self.add_model(version=1, promoted=True)
        second, _ = self.add_model(version=2)
        self.add_model(other=True)
        page = self.listing(page_size=1).json()
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["items"][0]["id"], str(second))
        self.assertFalse(page["items"][0]["is_promoted"])
        page = self.listing(page=2, page_size=1).json()
        self.assertEqual(page["items"][0]["id"], str(first))
        self.assertTrue(page["items"][0]["is_promoted"])
        self.assertEqual(self.listing(page=3, page_size=1).json()["items"], [])

    def test_detail_links_training_and_preserves_frozen_class_map(self):
        identifier, job = self.add_model()
        response = self.client.post(
            f"/api/v1/projects/{self.project_id}/classes",
            headers=self.headers,
            json={"classes": [{"name": "new defect", "color": "#ff0000"}]},
        )
        self.assertEqual(response.status_code, 201, response.text)
        response = self.detail(identifier)
        self.assertEqual(response.status_code, 200, response.text)
        row = response.json()
        self.assertEqual(row["job_id"], str(job))
        self.assertEqual(row["project_id"], str(self.project_id))
        self.assertEqual(row["dataset_version_id"], str(self.version_id))
        self.assertEqual(row["dataset_version"], 1)
        self.assertEqual(row["task_type"], "classification")
        self.assertEqual(row["model_arch"], "resnet18")
        self.assertEqual(row["input_size"], 224)
        self.assertEqual(row["class_map"], {"ok": 0, "defect": 1})
        self.assertEqual(row["hyperparams"], {"epochs": 5, "seed": 42})
        self.assertEqual(self.read(job).json()["model_id"], str(identifier))

    def test_detection_model_is_readable_by_its_owner(self):
        identifier, _ = self.add_model(other=True)
        response = self.detail(identifier, self.auth(self.other_id))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["task_type"], "detection")
        self.assertEqual(response.json()["model_arch"], "yolov8s")
        self.assertEqual(response.json()["input_size"], 640)

    def test_private_artifact_records_are_not_serialized(self):
        identifier, _ = self.add_model()
        for response in (self.listing(), self.detail(identifier)):
            self.assertEqual(response.headers["cache-control"], "no-store")
            for value in (
                "private-weights-key",
                "private-onnx-key",
                "do-not-expose-storage-records",
                '"weights_key"',
                '"onnx_key"',
                '"metrics"',
                '"artifacts"',
            ):
                self.assertNotIn(value, response.text)
        self.assertNotIn("class_map", self.listing().json()["items"][0])
        self.assertNotIn("hyperparams", self.listing().json()["items"][0])

    def test_owner_isolation_and_access_tokens_on_both_routes(self):
        identifier, _ = self.add_model()
        foreign, _ = self.add_model(other=True)
        for path in (
            f"/api/v1/models/{identifier}",
            f"/api/v1/projects/{self.project_id}/models",
        ):
            for headers in ({}, self.auth(self.user_id, "refresh")):
                self.assertEqual(
                    self.client.get(path, headers=headers).status_code, 401
                )
            self.assertEqual(
                self.client.get(path, headers=self.auth(self.other_id)).status_code, 404
            )
        self.assertEqual(self.detail(foreign).json(), self.detail(uuid4()).json())
        for project in (self.other_project_id, uuid4()):
            response = self.client.get(
                f"/api/v1/projects/{project}/models", headers=self.headers
            )
            self.assertEqual(response.status_code, 404)
        # Same owner, different project: lists must not leak across projects either.
        with Session(self.engine) as session:
            session.get(Project, self.other_project_id).owner_id = self.user_id
            session.commit()
        self.assertEqual(self.listing().json()["total"], 1)
        self.assertEqual(self.detail(foreign).status_code, 200)

    def test_pagination_and_identifier_validation(self):
        for params in (
            {"page": 0},
            {"page": 1_000_001},
            {"page_size": 0},
            {"page_size": 101},
        ):
            self.assertEqual(self.listing(**params).status_code, 422)
        self.assertEqual(self.detail("invalid").status_code, 422)

    def test_openapi_exposes_metadata_without_private_model_fields(self):
        schema = self.client.get("/openapi.json").json()
        for path in (
            "/api/v1/models/{model_id}",
            "/api/v1/projects/{project_id}/models",
        ):
            self.assertEqual(
                schema["paths"][path]["get"]["security"], [{"HTTPBearer": []}]
            )
        fields = schema["components"]["schemas"]["ModelDetail"]["properties"]
        self.assertIn("class_map", fields)
        self.assertIn("hyperparams", fields)
        self.assertNotIn("weights_key", fields)
        self.assertNotIn("metrics", fields)
