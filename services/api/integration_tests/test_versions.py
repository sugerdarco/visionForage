"""Frozen versions on real PostgreSQL and private MinIO buckets."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, BoundedSemaphore, Event
from unittest.mock import patch
from uuid import UUID, uuid4

from api_fixture import APITest
from botocore.exceptions import ClientError, EndpointConnectionError
import httpx2 as httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session
from test_dataset_snapshots import picture

from vf_api.dataset_snapshots import build_snapshot
from vf_api.models import Annotation, Class, DatasetVersion, Image, Project, User
from vf_api.routers.versions import slots
from vf_api.security import issue_token
from vf_api.settings import StorageSettings
from vf_api.storage import DatasetStore, ObjectStore
from vf_api.upload_dependencies import get_datasets, get_store


class VersionTest(APITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        suffix = uuid4().hex
        settings = StorageSettings(
            s3_images_bucket="vf-test-images-" + suffix,
            s3_datasets_bucket="vf-test-data-" + suffix,
        )
        cls.store = ObjectStore(settings)
        cls.datasets = DatasetStore(settings)
        for store in (cls.store, cls.datasets):
            cls.addClassCleanup(store.close)
            store.client.create_bucket(Bucket=store.bucket)
            cls.addClassCleanup(cls.drop_bucket, store)
        cls.app.dependency_overrides[get_store] = lambda: cls.store
        cls.app.dependency_overrides[get_datasets] = lambda: cls.datasets

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
        with Session(self.engine) as session:
            users = [
                User(email=f"{uuid4().hex}@example.com", password_hash="unused")
                for _ in range(2)
            ]
            session.add_all(users)
            session.flush()
            self.user_id, self.other_id = [row.id for row in users]
            session.commit()
        self.headers = self.auth(self.user_id)
        self.make_dataset()

    def auth(self, user_id, kind="access"):
        token, _ = issue_token(user_id, kind, self.settings)
        return {"Authorization": f"Bearer {token}"}

    def make_dataset(self, task="classification", count=12):
        self.images, self.keys = [], []
        with Session(self.engine) as session:
            project = Project(owner_id=self.user_id, name="Frozen data", task_type=task)
            session.add(project)
            session.flush()
            self.project_id = project.id
            classes = [
                Class(project_id=project.id, name=name, index=index, color="#123456")
                for index, name in enumerate(("ok", "defect"))
            ]
            session.add_all(classes)
            session.flush()
            self.class_ids = [row.id for row in classes]
            for index in range(count):
                data = picture(index)
                sha = hashlib.sha256(data).hexdigest()
                key = f"{project.id}/originals/{sha}.png"
                self.store.put(key, data, "image/png")
                image = Image(
                    project_id=project.id,
                    storage_key=key,
                    sha256=sha,
                    width=20,
                    height=10,
                    status="labeled",
                )
                session.add(image)
                session.flush()
                self.images.append(image.id)
                self.keys.append(key)
                indices = (index % 2,) if task == "classification" else (0, 1)
                for label in indices:
                    session.add(
                        Annotation(
                            image_id=image.id,
                            class_id=classes[label].id,
                            source="human",
                            bbox=None if task == "classification" else [0, 0, 0.5, 0.5],
                        )
                    )
            session.commit()
        self.path = f"/api/v1/projects/{self.project_id}/dataset-versions"

    def create(self, body=None):
        return self.client.post(self.path, headers=self.headers, json=body or {})

    def versions(self):
        with Session(self.engine) as session:
            return list(
                session.scalars(
                    select(DatasetVersion).where(
                        DatasetVersion.project_id == self.project_id
                    )
                )
            )

    def download(self, response):
        manifest = httpx.get(response["manifest_url"], timeout=15)
        audit = httpx.get(response["audit_url"], timeout=15)
        self.assertEqual(manifest.status_code, 200, manifest.text)
        self.assertEqual(audit.status_code, 200, audit.text)
        self.assertEqual(
            hashlib.sha256(manifest.content).hexdigest(),
            response["counts"]["manifest_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(audit.content).hexdigest(),
            response["counts"]["audit_sha256"],
        )
        return manifest, audit

    def test_create_private_artifacts_and_load_with_existing_training_contract(self):
        response = self.create()
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        row = response.json()
        self.assertEqual(row["version"], 1)
        self.assertEqual(row["counts"]["images"], {"train": 8, "val": 2, "test": 2})
        manifest, audit = self.download(row)
        self.assertEqual(manifest.json()["class_map"], {"ok": 0, "defect": 1})
        self.assertEqual(len(audit.json()["images"]), 12)
        self.assertIn("/" + self.datasets.bucket + "/", row["manifest_url"])
        for url in (row["manifest_url"], row["audit_url"]):
            self.assertEqual(
                httpx.get(url.split("?", 1)[0], timeout=15).status_code, 403
            )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_bytes(manifest.content)
            result = subprocess.run(
                [sys.executable, "-m", "vf_ml.manifest", str(path)],
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "packages"),
                },
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout)["images_by_split"], row["counts"]["images"]
            )

    def test_repeated_creation_is_a_new_version_with_reproducible_manifest(self):
        first = self.create().json()
        second = self.create().json()
        self.assertEqual((first["version"], second["version"]), (1, 2))
        self.assertNotEqual(first["manifest_key"], second["manifest_key"])
        self.assertEqual(
            first["counts"]["manifest_sha256"], second["counts"]["manifest_sha256"]
        )
        self.assertEqual(
            self.download(first)[0].content, self.download(second)[0].content
        )
        page = self.client.get(self.path + "?page_size=1", headers=self.headers).json()
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["items"][0]["id"], second["id"])
        page2 = self.client.get(
            self.path + "?page=2&page_size=1", headers=self.headers
        ).json()
        self.assertEqual(page2["items"][0]["id"], first["id"])
        detail = self.client.get(self.path + "/" + first["id"], headers=self.headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["counts"], first["counts"])

    def test_live_label_and_class_changes_do_not_modify_existing_versions(self):
        first = self.create().json()
        manifest, audit = self.download(first)
        edited = self.client.put(
            f"/api/v1/images/{self.images[0]}/annotations",
            headers=self.headers,
            json={"annotations": [{"class_id": str(self.class_ids[1])}]},
        )
        self.assertEqual(edited.status_code, 200)
        second = self.create().json()
        self.assertNotEqual(
            first["counts"]["manifest_sha256"], second["counts"]["manifest_sha256"]
        )
        self.client.post(
            f"/api/v1/projects/{self.project_id}/classes",
            headers=self.headers,
            json={"classes": [{"name": "new", "color": "#123456"}]},
        )
        frozen_manifest, frozen_audit = self.download(first)
        self.assertEqual(frozen_manifest.content, manifest.content)
        self.assertEqual(frozen_audit.content, audit.content)
        self.assertNotIn("new", frozen_manifest.json()["class_map"])
        with Session(self.engine) as session:
            row = session.get(DatasetVersion, UUID(first["id"]))
            row.version = 999
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_conditional_writes_refuse_to_replace_frozen_artifacts(self):
        first = self.create().json()
        before = self.download(first)[0].content
        for data in (before, b"different"):
            with self.assertRaises(ClientError) as failure:
                self.datasets.put_once(first["manifest_key"], data)
            self.assertEqual(
                failure.exception.response["ResponseMetadata"]["HTTPStatusCode"], 412
            )
        self.assertEqual(self.download(first)[0].content, before)

    def test_owner_scoping_validation_and_openapi(self):
        first = self.create().json()
        for method, path, body in (
            ("POST", self.path, {}),
            ("GET", self.path, None),
            ("GET", self.path + "/" + first["id"], None),
        ):
            for headers in ({}, self.auth(self.user_id, "refresh")):
                self.assertEqual(
                    self.client.request(
                        method, path, headers=headers, json=body
                    ).status_code,
                    401,
                )
            response = self.client.request(
                method, path, headers=self.auth(self.other_id), json=body
            )
            self.assertEqual(response.status_code, 404)
        self.assertEqual(
            self.client.get(
                self.path + "/" + str(uuid4()), headers=self.headers
            ).status_code,
            404,
        )
        self.make_dataset()
        self.assertEqual(
            self.client.get(
                self.path + "/" + first["id"], headers=self.headers
            ).status_code,
            404,
        )
        for body in (
            {"seed": True},
            {"seed": -1},
            {"manifest_key": "user-chosen"},
            {"split_ratios": {"train": 1, "val": 0, "test": 0}},
        ):
            self.assertEqual(self.create(body).status_code, 422)
        for query in ("page=0", "page_size=101", "page=bad"):
            self.assertEqual(
                self.client.get(
                    self.path + "?" + query, headers=self.headers
                ).status_code,
                422,
            )
        document = self.client.get("/openapi.json").json()
        for path, methods in (
            ("/api/v1/projects/{project_id}/dataset-versions", ("get", "post")),
            ("/api/v1/projects/{project_id}/dataset-versions/{version_id}", ("get",)),
        ):
            for method in methods:
                self.assertEqual(
                    document["paths"][path][method]["security"], [{"HTTPBearer": []}]
                )

    def test_excludes_unlabeled_and_review_images_and_accepts_detection_negatives(self):
        with Session(self.engine) as session:
            session.get(Image, self.images[0]).status = "unlabeled"
            session.get(Image, self.images[1]).status = "in_review"
            session.commit()
        first = self.create()
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(
            first.json()["counts"]["excluded"], {"unlabeled": 1, "in_review": 1}
        )
        self.assertEqual(sum(first.json()["counts"]["images"].values()), 10)
        self.make_dataset("detection")
        response = self.client.put(
            f"/api/v1/images/{self.images[0]}/annotations",
            headers=self.headers,
            json={"annotations": [], "status": "labeled"},
        )
        self.assertEqual(response.status_code, 200)
        frozen = self.create()
        self.assertEqual(frozen.status_code, 201, frozen.text)
        manifest = self.download(frozen.json())[0].json()
        self.assertEqual(
            next(
                row["labels"]
                for row in manifest["samples"]
                if row["image_key"] == self.keys[0]
            ),
            [],
        )

    def test_missing_corrupt_and_invalid_live_data_do_not_create_versions(self):
        self.store.delete(self.keys[0])
        self.assertEqual(self.create().status_code, 409)
        self.store.put(self.keys[0], b"corrupted", "image/png")
        self.assertEqual(self.create().status_code, 422)
        self.store.put(self.keys[0], picture(0), "image/png")
        with Session(self.engine) as session:
            annotation = session.scalar(
                select(Annotation).where(Annotation.image_id == self.images[0])
            )
            annotation.source, annotation.confidence = "model", 0.6
            session.commit()
        self.assertEqual(self.create().status_code, 422)
        self.assertEqual(self.versions(), [])

    def test_cross_project_labels_and_empty_classification_labels_are_rejected(self):
        other_project = self.project_id
        other_class = self.class_ids[0]
        self.make_dataset()
        self.assertNotEqual(other_project, self.project_id)
        with Session(self.engine) as session:
            annotation = session.scalar(
                select(Annotation).where(Annotation.image_id == self.images[0])
            )
            annotation.class_id = other_class
            session.commit()
        self.assertEqual(self.create().status_code, 422)
        with Session(self.engine) as session:
            annotation = session.scalar(
                select(Annotation).where(Annotation.image_id == self.images[0])
            )
            session.delete(annotation)
            session.commit()
        self.assertEqual(self.create().status_code, 422)
        self.assertEqual(self.versions(), [])

    def test_storage_failure_does_not_commit_and_retry_uses_a_new_attempt_namespace(
        self,
    ):
        original = self.datasets.put_once
        written = []

        def fail_audit(key, data):
            if key.endswith("audit.json"):
                raise EndpointConnectionError(endpoint_url="http://internal-only")
            written.append(key)
            original(key, data)

        with patch.object(self.datasets, "put_once", side_effect=fail_audit):
            response = self.create()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": "Object storage unavailable"})
        self.assertEqual(self.versions(), [])
        retry = self.create()
        self.assertEqual(retry.status_code, 201, retry.text)
        self.assertEqual(retry.json()["version"], 1)
        self.assertNotEqual(retry.json()["manifest_key"], written[0])
        self.assertEqual(
            self.datasets.read(written[0]), self.download(retry.json())[0].content
        )

    def test_database_commit_failure_leaves_no_record_and_retry_succeeds(self):
        with patch.object(
            Session,
            "commit",
            side_effect=OperationalError("COMMIT", {}, Exception("unavailable")),
        ):
            response = self.create()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": "Database unavailable"})
        self.assertEqual(self.versions(), [])
        self.assertEqual(self.create().status_code, 201)

    def test_metadata_is_frozen_and_editing_locks_are_released_before_image_reads(self):
        reading, resume = Event(), Event()
        original = self.store.read
        first_read = True

        def pause_read(key, **kwargs):
            nonlocal first_read
            if first_read:
                first_read = False
                reading.set()
                if not resume.wait(timeout=10):
                    raise AssertionError("Image reader was never resumed")
            return original(key, **kwargs)

        with patch.object(self.store, "read", side_effect=pause_read):
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = pool.submit(self.create)
                try:
                    self.assertTrue(reading.wait(timeout=10))
                    edit = pool.submit(
                        self.client.put,
                        f"/api/v1/images/{self.images[0]}/annotations",
                        headers=self.headers,
                        json={"annotations": [{"class_id": str(self.class_ids[1])}]},
                    )
                    self.assertEqual(edit.result(timeout=5).status_code, 200)
                finally:
                    resume.set()
                response = pending.result(timeout=10)
        self.assertEqual(response.status_code, 201, response.text)
        manifest = self.download(response.json())[0].json()
        row = next(
            row for row in manifest["samples"] if row["image_key"] == self.keys[0]
        )
        self.assertEqual(row["labels"][0]["class_index"], 0)

    def test_busy_response_and_concurrent_processes_allocate_distinct_versions(self):
        self.assertTrue(slots.acquire(blocking=False))
        try:
            response = self.create()
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["retry-after"], "5")
        finally:
            slots.release()
        barrier = Barrier(2)

        def ready(*args, **kwargs):
            result = build_snapshot(*args, **kwargs)
            barrier.wait(timeout=10)
            return result

        # Two slots simulate separate API processes reaching final registration.
        with (
            patch("vf_api.routers.versions.slots", BoundedSemaphore(2)),
            patch("vf_api.routers.versions.build_snapshot", side_effect=ready),
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda _: self.create(), range(2)))
        self.assertEqual([row.status_code for row in responses], [201, 201])
        self.assertEqual(sorted(row.json()["version"] for row in responses), [1, 2])
        self.assertEqual(len({row.json()["manifest_key"] for row in responses}), 2)

    def test_metadata_limits_and_no_labeled_images_fail_before_storage_reads(self):
        for name in ("MAX_CLASSES", "MAX_IMAGES", "MAX_LABELS"):
            with (
                patch("vf_api.dataset_snapshots." + name, 1),
                patch.object(self.store, "read", wraps=self.store.read) as read,
            ):
                response = self.create()
                self.assertEqual(response.status_code, 422, response.text)
                read.assert_not_called()
        with Session(self.engine) as session:
            for image in session.scalars(
                select(Image).where(Image.project_id == self.project_id)
            ):
                image.status = "unlabeled"
            session.commit()
        response = self.create()
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("Label images", response.json()["detail"])
        self.assertEqual(self.versions(), [])
