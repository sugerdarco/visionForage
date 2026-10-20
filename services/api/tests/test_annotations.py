"""Label ownership, validation and concurrent replacement on real PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace
from uuid import UUID, uuid4

from api_fixture import APITest
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.models import Annotation, Class, Image, Project, User
from vf_api.security import issue_token
from vf_api.upload_dependencies import get_store


class AnnotationTest(APITest):
    def setUp(self):
        with Session(self.engine) as session:
            owner = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            other = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            session.add_all([owner, other])
            session.flush()
            self.owner_id, self.other_id = owner.id, other.id
            session.commit()
        self.headers = self.auth(self.owner_id)
        self.project_id, self.image_id, self.classes = self.make_image()
        self.path = f"/api/v1/images/{self.image_id}/annotations"

    def auth(self, user_id, kind="access"):
        token, _ = issue_token(user_id, kind, self.settings)
        return {"Authorization": f"Bearer {token}"}

    def make_image(self, task="classification", owner_id=None):
        with Session(self.engine) as session:
            project = Project(
                owner_id=owner_id or self.owner_id, name="Inspection", task_type=task
            )
            session.add(project)
            session.flush()
            image = Image(
                project_id=project.id,
                storage_key=f"test/{uuid4()}",
                width=200,
                height=100,
                sha256=uuid4().hex * 2,
            )
            classes = [
                Class(project_id=project.id, name=name, color="#123456", index=index)
                for index, name in enumerate(("ok", "defect"))
            ]
            session.add_all([image, *classes])
            session.flush()
            result = project.id, image.id, [str(row.id) for row in classes]
            session.commit()
            return result

    def save(self, annotations, *, image_id=None, **fields):
        return self.client.put(
            f"/api/v1/images/{image_id or self.image_id}/annotations",
            headers=self.headers,
            json={"annotations": annotations, **fields},
        )

    def read(self, image_id=None):
        response = self.client.get(
            f"/api/v1/images/{image_id or self.image_id}/annotations",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_classification_save_replace_clear_and_retry(self):
        self.assertEqual(self.read()["annotations"], [])
        first = self.save([{"class_id": self.classes[0]}])
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.headers["cache-control"], "no-store")
        self.assertEqual(first.json()["status"], "labeled")
        label = first.json()["annotations"][0]
        self.assertEqual(label["source"], "human")
        self.assertIsNone(label["confidence"])
        self.assertIsNone(label["bbox"])
        with Session(self.engine) as session:
            before = session.get(Image, self.image_id).updated_at
            annotation_before = session.get(Annotation, UUID(label["id"])).updated_at
        retry = self.save([{"class_id": self.classes[0]}])
        self.assertEqual(retry.json(), first.json())
        with Session(self.engine) as session:
            self.assertEqual(session.get(Image, self.image_id).updated_at, before)
            self.assertEqual(
                session.get(Annotation, UUID(label["id"])).updated_at,
                annotation_before,
            )
        replaced = self.save([{"class_id": self.classes[1]}]).json()
        self.assertEqual(len(replaced["annotations"]), 1)
        self.assertEqual(replaced["annotations"][0]["class_id"], self.classes[1])
        self.assertEqual(self.read(), replaced)
        cleared = self.save([]).json()
        self.assertEqual(cleared["status"], "unlabeled")
        self.assertEqual(cleared["annotations"], [])
        self.assertEqual(self.save([]).json(), cleared)

    def test_detection_multiple_boxes_order_independent_retry_and_negative(self):
        _, image_id, classes = self.make_image("detection")
        labels = [
            {"class_id": classes[0], "bbox": [0, 0, 1, 1]},
            {"class_id": classes[0], "bbox": [0.2, 0.3, 0.4, 0.5]},
            {"class_id": classes[1], "bbox": [0.5, 0.5, 0.5, 0.5]},
        ]
        first = self.save(labels, image_id=image_id)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(len(first.json()["annotations"]), 3)
        self.assertEqual(
            self.save(list(reversed(labels)), image_id=image_id).json(), first.json()
        )
        self.assertEqual(self.read(image_id), first.json())
        negative = self.save([], image_id=image_id, status="labeled")
        self.assertEqual(negative.status_code, 200)
        self.assertEqual(negative.json()["status"], "labeled")
        self.assertEqual(negative.json()["annotations"], [])
        self.assertEqual(self.read(image_id), negative.json())
        self.assertEqual(self.save([], image_id=image_id).json()["status"], "unlabeled")

    def test_ownership_missing_images_and_authentication(self):
        first = self.save([{"class_id": self.classes[0]}]).json()
        for method in ("GET", "PUT"):
            for image_id in (self.image_id, uuid4()):
                response = self.client.request(
                    method,
                    f"/api/v1/images/{image_id}/annotations",
                    headers=self.auth(self.other_id),
                    json={"annotations": []} if method == "PUT" else None,
                )
                self.assertEqual(response.status_code, 404, response.text)
                self.assertEqual(response.json(), {"detail": "Image not found"})
            for headers in ({}, self.auth(self.owner_id, "refresh")):
                response = self.client.request(
                    method,
                    self.path,
                    headers=headers,
                    json={"annotations": []} if method == "PUT" else None,
                )
                self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.read(), first)

    def test_wrong_project_classes_are_rejected_without_partial_changes(self):
        first = self.save([{"class_id": self.classes[0]}]).json()
        _, _, same_owner_classes = self.make_image()
        _, _, other_owner_classes = self.make_image(owner_id=self.other_id)
        errors = []
        for class_id in (same_owner_classes[0], other_owner_classes[0], str(uuid4())):
            response = self.save([{"class_id": class_id}])
            self.assertEqual(response.status_code, 422, response.text)
            errors.append(response.json())
            self.assertEqual(self.read(), first)
        self.assertTrue(all(error == errors[0] for error in errors))
        _, image_id, classes = self.make_image("detection")
        good = {"class_id": classes[0], "bbox": [0, 0, 1, 1]}
        saved = self.save([good], image_id=image_id).json()
        response = self.save(
            [good, {"class_id": self.classes[0], "bbox": [0, 0, 1, 1]}],
            image_id=image_id,
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.read(image_id), saved)

    def test_task_rules_and_client_controlled_fields(self):
        first = self.save([{"class_id": self.classes[0]}]).json()
        for body in (
            {},
            {"annotations": None},
            {"annotations": [], "status": "labeled"},
            {"annotations": [], "status": "in_review"},
            {"annotations": [], "image_id": str(self.image_id)},
            {"annotations": [{"class_id": value} for value in self.classes]},
            {"annotations": [{"class_id": self.classes[0], "bbox": [0, 0, 1, 1]}]},
            {"annotations": [{"class_id": self.classes[0]}], "status": "unlabeled"},
            *(
                {"annotations": [{"class_id": self.classes[0], key: value}]}
                for key, value in (
                    ("source", "model"),
                    ("confidence", 0.9),
                    ("id", str(uuid4())),
                    ("image_id", str(uuid4())),
                )
            ),
        ):
            response = self.client.put(self.path, headers=self.headers, json=body)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.read(), first)
        _, image_id, classes = self.make_image("detection")
        self.assertEqual(
            self.save([{"class_id": classes[0]}], image_id=image_id).status_code, 422
        )
        _, image_id, _ = self.make_image("anomaly")
        self.assertEqual(self.save([], image_id=image_id).status_code, 422)

    def test_invalid_boxes_leave_existing_labels_intact(self):
        _, image_id, classes = self.make_image("detection")
        label = {"class_id": classes[0], "bbox": [0, 0, 1, 1]}
        first = self.save([label], image_id=image_id).json()
        for box in (
            [],
            [0, 0, 1],
            [0, 0, 1, 1, 1],
            [[0], 0, 1, 1],
            [-0.1, 0, 1, 1],
            [0, 1.1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 0, 1.1, 1],
            [0.2, 0, 0.9, 1],
            [0, 0.2, 1, 0.9],
            [None, 0, 1, 1],
            ["0", 0, 1, 1],
            [False, 0, 1, 1],
            [float("nan"), 0, 1, 1],
            [0, 0, float("inf"), 1],
        ):
            with self.subTest(box=box):
                response = self.client.put(
                    f"/api/v1/images/{image_id}/annotations",
                    headers={**self.headers, "Content-Type": "application/json"},
                    content=json.dumps({"annotations": [{**label, "bbox": box}]}),
                )
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(self.read(image_id), first)
        # Normal decimal addition at the right/bottom edge remains valid.
        edge = self.save([{**label, "bbox": [0.1, 0.1, 0.9, 0.9]}], image_id=image_id)
        self.assertEqual(edge.status_code, 200, edge.text)

    def test_bounded_batch_preserves_repeated_boxes(self):
        _, image_id, classes = self.make_image("detection")
        label = {"class_id": classes[0], "bbox": [0, 0, 1, 1]}
        response = self.save([label] * 1000, image_id=image_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["annotations"]), 1000)
        self.assertEqual(self.save([label] * 1001, image_id=image_id).status_code, 422)
        self.assertEqual(self.read(image_id), response.json())

    def test_human_save_replaces_model_suggestions_and_review_status(self):
        with Session(self.engine) as session:
            session.get(Image, self.image_id).status = "in_review"
            session.add(
                Annotation(
                    image_id=self.image_id,
                    class_id=UUID(self.classes[0]),
                    source="model",
                    confidence=0.7,
                )
            )
            session.commit()
        self.assertEqual(self.read()["status"], "in_review")
        response = self.save([{"class_id": self.classes[0]}])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "labeled")
        row = response.json()["annotations"][0]
        self.assertEqual(row["source"], "human")
        self.assertIsNone(row["confidence"])

    def test_concurrent_replacements_never_merge_and_identical_retries_keep_ids(self):
        _, image_id, classes = self.make_image("detection")
        barrier = Barrier(4)
        batches = [
            [{"class_id": classes[index % 2], "bbox": [0, 0, (index + 1) / 4, 1]}]
            * (index + 1)
            for index in range(4)
        ]

        def submit(batch):
            barrier.wait(timeout=10)
            return self.save(batch, image_id=image_id)

        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(submit, batches))
        for batch, response in zip(batches, responses):
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()["annotations"]), len(batch))
            for row in response.json()["annotations"]:
                self.assertEqual(row["class_id"], batch[0]["class_id"])
                self.assertEqual(row["bbox"], batch[0]["bbox"])
        final = self.read(image_id)
        self.assertIn(final, [response.json() for response in responses])
        with ThreadPoolExecutor(max_workers=4) as pool:
            retries = list(pool.map(submit, [batches[0]] * 4))
        self.assertEqual([response.status_code for response in retries], [200] * 4)
        self.assertTrue(all(row.json() == retries[0].json() for row in retries))
        self.assertEqual(self.read(image_id), retries[0].json())

    def test_gallery_filters_follow_saved_labels_and_negative_status(self):
        self.app.dependency_overrides[get_store] = lambda: SimpleNamespace(
            url=lambda key: f"https://storage.invalid/{key}"
        )
        self.addCleanup(self.app.dependency_overrides.pop, get_store)
        project_id, image_id, classes = self.make_image("detection")
        path = f"/api/v1/projects/{project_id}/images"
        label = {"class_id": classes[0], "bbox": [0, 0, 1, 1]}
        self.save([label, label], image_id=image_id)
        response = self.client.get(
            path,
            params={"class": classes[0], "status": "labeled"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 1)
        self.save([], image_id=image_id, status="labeled")
        self.assertEqual(
            self.client.get(
                path, params={"class": classes[0]}, headers=self.headers
            ).json()["total"],
            0,
        )
        self.assertEqual(
            self.client.get(
                path, params={"status": "labeled"}, headers=self.headers
            ).json()["total"],
            1,
        )
        self.save([], image_id=image_id)
        self.assertEqual(
            self.client.get(
                path, params={"status": "unlabeled"}, headers=self.headers
            ).json()["total"],
            1,
        )
        with Session(self.engine) as session:
            self.assertEqual(
                list(
                    session.scalars(
                        select(Annotation).where(Annotation.image_id == image_id)
                    )
                ),
                [],
            )

    def test_openapi_documents_read_save_and_bearer_security(self):
        document = self.client.get("/openapi.json").json()
        path = document["paths"]["/api/v1/images/{image_id}/annotations"]
        for method in ("get", "put"):
            self.assertEqual(path[method]["security"], [{"HTTPBearer": []}])
        schema = document["components"]["schemas"]["AnnotationSave"]
        self.assertEqual(schema["properties"]["annotations"]["maxItems"], 1000)
        self.assertFalse(schema["additionalProperties"])
