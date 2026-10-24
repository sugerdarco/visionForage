"""Project isolation and class-map transactions on real PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
import json
from uuid import UUID, uuid4

from api_fixture import APITest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from vf_api.models import (
    Class,
    DatasetVersion,
    Image,
    Model,
    Project,
    TrainingJob,
    User,
)
from vf_api.security import issue_token

PREFIX = "/api/v1/projects"


class ProjectTest(APITest):
    def setUp(self):
        with Session(self.engine) as session:
            self.owner = User(
                email=f"{uuid4().hex}@example.com", password_hash="unused"
            )
            self.other = User(
                email=f"{uuid4().hex}@example.com", password_hash="unused"
            )
            session.add_all([self.owner, self.other])
            session.flush()
            self.owner_id, self.other_id = self.owner.id, self.other.id
            session.commit()
        self.headers = self.auth(self.owner_id)
        self.other_headers = self.auth(self.other_id)

    def auth(self, user_id):
        token, _ = issue_token(user_id, "access", self.settings)
        return {"Authorization": f"Bearer {token}"}

    def create(
        self, name="Inspection", task_type="classification", classes=None, headers=None
    ):
        response = self.client.post(
            PREFIX,
            headers=headers or self.headers,
            json={"name": name, "task_type": task_type, "classes": classes or []},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def labels(self, project_id, names, headers=None):
        return self.client.post(
            f"{PREFIX}/{project_id}/classes",
            headers=headers or self.headers,
            json={"classes": [{"name": name, "color": "#00AAFF"} for name in names]},
        )

    def test_create_read_rename_and_delete_project_with_classes(self):
        project = self.create(
            name="  Casting  ",
            classes=[
                {"name": " ok ", "color": "#00AAFF"},
                {"name": "defect", "color": "#ff0000"},
            ],
        )
        self.assertEqual(project["name"], "Casting")
        self.assertEqual(project["owner_id"], str(self.owner_id))
        self.assertEqual([row["index"] for row in project["classes"]], [0, 1])
        self.assertEqual(project["classes"][0]["name"], "ok")
        self.assertEqual(project["classes"][0]["color"], "#00aaff")
        path = f"{PREFIX}/{project['id']}"
        self.assertEqual(self.client.get(path, headers=self.headers).json(), project)
        renamed = self.client.patch(
            path, headers=self.headers, json={"name": "Casting v2"}
        )
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()["name"], "Casting v2")
        self.assertGreater(renamed.json()["updated_at"], project["updated_at"])
        self.assertEqual(renamed.json()["classes"], project["classes"])
        deleted = self.client.delete(path, headers=self.headers)
        self.assertEqual(deleted.status_code, 202)
        self.assertEqual(deleted.json()["status"], "deleting")
        self.assertTrue(self.client.get(path, headers=self.headers).json()["deleting"])
        self.assertEqual(
            self.client.delete(path, headers=self.headers).status_code, 202
        )
        self.assertEqual(
            self.client.patch(
                path, headers=self.headers, json={"name": "Late edit"}
            ).status_code,
            409,
        )
        with Session(self.engine) as session:
            self.assertEqual(
                session.scalar(
                    sa.select(sa.func.count())
                    .select_from(Class)
                    .where(Class.project_id == UUID(project["id"]))
                ),
                len(project["classes"]),
            )

    def test_owner_scoped_pagination_and_deterministic_order(self):
        ids = [self.create(name=f"Project {number}")["id"] for number in range(3)]
        self.create(headers=self.other_headers)
        first = self.client.get(PREFIX + "?page_size=2", headers=self.headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers["cache-control"], "no-store")
        self.assertEqual(first.json()["total"], 3)
        self.assertEqual(first.json()["page"], 1)
        self.assertEqual(first.json()["page_size"], 2)
        self.assertEqual([row["id"] for row in first.json()["items"]], ids[::-1][:2])
        second = self.client.get(
            PREFIX + "?page=2&page_size=2", headers=self.headers
        ).json()
        self.assertEqual([row["id"] for row in second["items"]], ids[:1])
        empty = self.client.get(
            PREFIX + "?page=3&page_size=2", headers=self.headers
        ).json()
        self.assertEqual(empty["items"], [])
        self.assertEqual(empty["total"], 3)
        # Equal creation times still have a stable UUID tie-breaker.
        with Session(self.engine) as session:
            session.execute(
                sa.update(Project)
                .where(Project.owner_id == self.owner_id)
                .values(created_at="2026-01-01T00:00:00Z")
            )
            session.commit()
        tied = self.client.get(PREFIX, headers=self.headers).json()["items"]
        self.assertEqual([row["id"] for row in tied], sorted(ids, reverse=True))

    def test_foreign_and_missing_projects_share_404_for_every_route(self):
        project = self.create(classes=[{"name": "original", "color": "#123456"}])
        requests = [
            ("GET", "", None),
            ("PATCH", "", {"name": "stolen"}),
            ("DELETE", "", None),
            ("GET", "/classes", None),
            (
                "POST",
                "/classes",
                {"classes": [{"name": "intruder", "color": "#ffffff"}]},
            ),
        ]
        for method, suffix, body in requests:
            for project_id in (project["id"], str(uuid4())):
                with self.subTest(method=method, suffix=suffix):
                    response = self.client.request(
                        method,
                        f"{PREFIX}/{project_id}{suffix}",
                        headers=self.other_headers,
                        json=body,
                    )
                    self.assertEqual(response.status_code, 404, response.text)
                    self.assertEqual(response.json(), {"detail": "Project not found"})
                    self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(
            self.client.get(f"{PREFIX}/{project['id']}", headers=self.headers).json(),
            project,
        )
        self.assertEqual(
            self.client.get(PREFIX, headers=self.other_headers).json()["total"], 0
        )

    def test_all_routes_require_access_tokens(self):
        project_id = self.create()["id"]
        refresh, _ = issue_token(self.owner_id, "refresh", self.settings)
        routes = [
            ("GET", PREFIX, None),
            ("POST", PREFIX, {"name": "Test", "task_type": "detection"}),
            ("GET", f"{PREFIX}/{project_id}", None),
            ("PATCH", f"{PREFIX}/{project_id}", {"name": "Renamed"}),
            ("DELETE", f"{PREFIX}/{project_id}", None),
            ("GET", f"{PREFIX}/{project_id}/classes", None),
            (
                "POST",
                f"{PREFIX}/{project_id}/classes",
                {"classes": [{"name": "a", "color": "#112233"}]},
            ),
        ]
        for method, path, body in routes:
            for headers in ({}, {"Authorization": f"Bearer {refresh}"}):
                response = self.client.request(method, path, headers=headers, json=body)
                self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(
            self.client.get(f"{PREFIX}/{project_id}", headers=self.headers).status_code,
            200,
        )

    def test_invalid_inputs_and_immutable_fields(self):
        for body in (
            {"name": " ", "task_type": "classification"},
            {"name": "x" * 201, "task_type": "classification"},
            {"name": "bad\0name", "task_type": "classification"},
            {"name": "Anomaly", "task_type": "anomaly"},
            {"name": "Other", "task_type": "detection", "owner_id": str(self.other_id)},
            {
                "name": "Other",
                "task_type": "detection",
                "classes": [{"name": "a", "color": "red"}],
            },
            {
                "name": "Other",
                "task_type": "detection",
                "classes": [{"name": "a", "color": "#ffffff", "index": 4}],
            },
            {
                "name": "Other",
                "task_type": "detection",
                "classes": [
                    {"name": " a ", "color": "#ffffff"},
                    {"name": "a", "color": "#000000"},
                ],
            },
        ):
            response = self.client.post(PREFIX, headers=self.headers, json=body)
            self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post(
            PREFIX,
            headers={**self.headers, "Content-Type": "application/json"},
            content=json.dumps({"name": "invalid-\ud800", "task_type": "detection"}),
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self.client.get(PREFIX, headers=self.headers).json()["total"], 0
        )
        project = self.create(task_type="detection")
        path = f"{PREFIX}/{project['id']}"
        for body in (
            {},
            {"name": None},
            {"task_type": "classification"},
            {"owner_id": str(self.other_id)},
            {"name": "Rename", "classes": []},
        ):
            self.assertEqual(
                self.client.patch(path, headers=self.headers, json=body).status_code,
                422,
            )
        self.assertEqual(self.client.get(path, headers=self.headers).json(), project)
        for query in (
            "page=0",
            "page=1000001",
            "page_size=0",
            "page_size=101",
            "page=abc",
        ):
            self.assertEqual(
                self.client.get(PREFIX + "?" + query, headers=self.headers).status_code,
                422,
            )
        self.assertEqual(
            self.client.get(PREFIX + "/not-a-uuid", headers=self.headers).status_code,
            422,
        )

    def test_class_batches_are_atomic_and_indices_are_server_assigned(self):
        project_id = self.create()["id"]
        path = f"{PREFIX}/{project_id}/classes"
        first = self.labels(project_id, ["a", "b"])
        self.assertEqual(first.status_code, 201)
        self.assertEqual([row["index"] for row in first.json()], [0, 1])
        self.assertEqual(self.labels(project_id, ["c", "a"]).status_code, 409)
        self.assertEqual(
            self.client.get(path, headers=self.headers).json(), first.json()
        )
        second = self.labels(project_id, ["c"])
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.json()[0]["index"], 2)
        for classes in (
            [],
            [{"name": "d", "color": "#ffffff", "index": 0}],
            [{"name": "d", "color": "#ffffff", "project_id": str(uuid4())}],
            [{"name": "d", "color": "#fff"}],
            [{"name": "d", "color": "#ffffff"}, {"name": " d ", "color": "#000000"}],
            [{"name": f"label-{i}", "color": "#ffffff"} for i in range(101)],
        ):
            response = self.client.post(
                path, headers=self.headers, json={"classes": classes}
            )
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(len(self.client.get(path, headers=self.headers).json()), 3)

    def test_concurrent_appends_allocate_distinct_contiguous_indices(self):
        project_id = self.create()["id"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(
                pool.map(
                    lambda n: self.labels(project_id, [f"a{n}", f"b{n}"]), range(4)
                )
            )
        self.assertEqual([response.status_code for response in responses], [201] * 4)
        rows = self.client.get(
            f"{PREFIX}/{project_id}/classes", headers=self.headers
        ).json()
        self.assertEqual([row["index"] for row in rows], list(range(8)))
        self.assertEqual(len({row["name"] for row in rows}), 8)
        with ThreadPoolExecutor(max_workers=2) as pool:
            duplicates = list(
                pool.map(lambda _: self.labels(project_id, ["same"]), range(2))
            )
        self.assertEqual(
            sorted(response.status_code for response in duplicates), [201, 409]
        )
        self.assertEqual(self.labels(project_id, ["next"]).json()[0]["index"], 9)

    def test_delete_with_image_marks_cleanup_without_removing_metadata(self):
        project = self.create(classes=[{"name": "keep", "color": "#123456"}])
        project_id = UUID(project["id"])
        with Session(self.engine) as session:
            session.add(
                Image(
                    project_id=project_id,
                    storage_key=f"test/{uuid4()}",
                    width=10,
                    height=10,
                    sha256="a" * 64,
                )
            )
            session.commit()
        response = self.client.delete(f"{PREFIX}/{project_id}", headers=self.headers)
        self.assertEqual(response.status_code, 202, response.text)
        current = self.client.get(f"{PREFIX}/{project_id}", headers=self.headers).json()
        self.assertTrue(current["deleting"])
        self.assertEqual(current["classes"], project["classes"])

    def test_appending_after_training_preserves_existing_class_map_and_history(self):
        project = self.create(classes=[{"name": "ok", "color": "#00ff00"}])
        project_id = UUID(project["id"])
        with Session(self.engine) as session:
            version = DatasetVersion(
                project_id=project_id,
                version=1,
                manifest_key=f"test/{uuid4()}",
                split_ratios={"train": 0.8, "val": 0.2},
                counts={"train": 8, "val": 2},
            )
            session.add(version)
            session.flush()
            job = TrainingJob(
                project_id=project_id,
                dataset_version_id=version.id,
                model_arch="resnet18",
                status="succeeded",
            )
            session.add(job)
            session.flush()
            model = Model(
                project_id=project_id,
                job_id=job.id,
                version=1,
                metrics={},
                weights_key="test/weights",
                onnx_key="test/onnx",
                class_map={"0": "ok"},
                input_size=224,
            )
            session.add(model)
            session.flush()
            model_id = model.id
            session.commit()
        response = self.labels(project_id, ["defect"])
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()[0]["index"], 1)
        rows = self.client.get(
            f"{PREFIX}/{project_id}/classes", headers=self.headers
        ).json()
        self.assertEqual(rows[0], project["classes"][0])
        with Session(self.engine) as session:
            self.assertEqual(session.get(Model, model_id).class_map, {"0": "ok"})
        self.assertEqual(
            self.client.delete(
                f"{PREFIX}/{project_id}", headers=self.headers
            ).status_code,
            202,
        )
        self.assertEqual(
            self.client.get(f"{PREFIX}/{project_id}", headers=self.headers).json()[
                "classes"
            ],
            rows,
        )

    def test_openapi_includes_project_routes_and_security(self):
        paths = self.client.get("/openapi.json").json()["paths"]
        for path, methods in (
            (PREFIX, ("get", "post")),
            (PREFIX + "/{project_id}", ("get", "patch", "delete")),
            (PREFIX + "/{project_id}/classes", ("get", "post")),
        ):
            for method in methods:
                self.assertEqual(paths[path][method]["security"], [{"HTTPBearer": []}])
