"""Signed dataset archives, database transactions and CPU thumbnails with real services."""

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
from threading import Event
import time
from unittest.mock import patch
from uuid import UUID, uuid4

from api_fixture import APITest
from botocore.exceptions import EndpointConnectionError
from celery.contrib.testing.worker import start_worker
import httpx2 as httpx
from import_fixture import claim_for, dataset_files, picture, zipped
import jwt
from kombu.exceptions import OperationalError as QueueError
from redis import Redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.import_tokens import read_import
from vf_api.dependencies import owned_project
from vf_api.models import Annotation, Class, Image, User
from vf_api.routers.imports import slots
from vf_api.security import issue_token
from vf_api.settings import QueueSettings, StorageSettings
from vf_api.storage import ObjectStore
from vf_api.upload_dependencies import get_queue, get_store
from vf_worker.app import create_worker


class ImportTest(APITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        suffix = uuid4().hex
        cls.storage_settings = StorageSettings(s3_images_bucket="vf-test-" + suffix)
        cls.store = ObjectStore(cls.storage_settings)
        cls.addClassCleanup(cls.store.close)
        cls.store.client.create_bucket(Bucket=cls.store.bucket)
        cls.addClassCleanup(cls.drop_bucket)
        cls.queue_settings = QueueSettings()
        cls.worker = create_worker(
            cls.settings, cls.storage_settings, cls.queue_settings
        )
        cls.addClassCleanup(cls.worker.close)
        cls.prefix = "vf-test-" + suffix + ":"
        cls.worker.conf.broker_transport_options = {
            **cls.worker.conf.broker_transport_options,
            "global_keyprefix": cls.prefix,
        }
        cls.worker.conf.task_default_queue = "vf-test-" + suffix
        cls.addClassCleanup(cls.drop_queue)
        cls.app.dependency_overrides[get_store] = lambda: cls.store
        cls.app.dependency_overrides[get_queue] = lambda: cls.worker
        consumer = start_worker(
            cls.worker,
            perform_ping_check=False,
            pool="solo",
            concurrency=1,
            loglevel="ERROR",
            shutdown_timeout=15,
        )
        consumer.__enter__()
        cls.addClassCleanup(consumer.__exit__, None, None, None)

    @classmethod
    def drop_bucket(cls):
        for page in cls.store.client.get_paginator("list_objects_v2").paginate(
            Bucket=cls.store.bucket
        ):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                cls.store.client.delete_objects(
                    Bucket=cls.store.bucket, Delete={"Objects": objects}
                )
        cls.store.client.delete_bucket(Bucket=cls.store.bucket)

    @classmethod
    def drop_queue(cls):
        with Redis.from_url(cls.queue_settings.broker_url) as client:
            keys = list(client.scan_iter(match=cls.prefix + "*"))
            if keys:
                client.delete(*keys)

    def setUp(self):
        with Session(self.engine) as session:
            user = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            other = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            session.add_all([user, other])
            session.flush()
            self.user_id, self.other_id = user.id, other.id
            session.commit()
        self.headers = self.auth(self.user_id)
        self.project_id = self.project("classification")

    def auth(self, user_id, kind="access"):
        token, _ = issue_token(user_id, kind, self.settings)
        return {"Authorization": f"Bearer {token}"}

    def project(self, task):
        response = self.client.post(
            "/api/v1/projects",
            headers=self.headers,
            json={"name": "Import test", "task_type": task},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def permission(self, data, format="imagefolder", **options):
        response = self.client.post(
            f"/api/v1/projects/{self.project_id}/imports/presign",
            headers=self.headers,
            json=claim_for(data, format, **options).model_dump(),
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def upload(self, files=None, format="imagefolder", **options):
        data = zipped(files or dataset_files(format))
        grant = self.permission(data, format, **options)
        response = httpx.put(
            grant["upload_url"], headers=grant["headers"], content=data, timeout=15
        )
        self.assertEqual(response.status_code, 200, response.text)
        return grant

    def complete(self, grant, headers=None):
        return self.client.post(
            f"/api/v1/projects/{self.project_id}/imports",
            headers=headers or self.headers,
            json={"upload_token": grant["upload_token"]},
        )

    def rows(self, model):
        with Session(self.engine) as session:
            return list(
                session.scalars(
                    select(model).where(model.project_id == UUID(self.project_id))
                )
            )

    def annotations(self, image_id):
        response = self.client.get(
            f"/api/v1/images/{image_id}/annotations", headers=self.headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_all_formats_signed_upload_registration_private_objects_and_thumbnails(
        self,
    ):
        for format in ("imagefolder", "yolo", "coco"):
            with self.subTest(format=format):
                self.project_id = self.project(
                    "classification" if format == "imagefolder" else "detection"
                )
                grant = self.upload(format=format)
                response = self.complete(grant)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["cache-control"], "no-store")
                result = response.json()
                self.assertEqual(result["created_images"], 1)
                self.assertEqual(result["created_classes"], 1)
                image_id = result["items"][0]["image_id"]
                labels = self.annotations(image_id)
                self.assertEqual(labels["status"], "labeled")
                if format == "imagefolder":
                    self.assertIsNone(labels["annotations"][0]["bbox"])
                else:
                    for actual, expected in zip(
                        labels["annotations"][0]["bbox"], [0.1, 0.1, 0.4, 0.4]
                    ):
                        self.assertAlmostEqual(actual, expected)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    image = self.client.get(
                        f"/api/v1/images/{image_id}", headers=self.headers
                    ).json()
                    if image["thumbnail_url"]:
                        break
                    time.sleep(0.1)
                self.assertTrue(image["thumbnail_url"])
                self.assertEqual(
                    httpx.get(image["image_url"], timeout=15).content, picture()
                )
                self.assertEqual(
                    httpx.get(
                        image["image_url"].split("?", 1)[0], timeout=15
                    ).status_code,
                    403,
                )
                retry = self.complete(grant)
                self.assertEqual(retry.status_code, 200, retry.text)
                self.assertEqual(retry.json()["unchanged_images"], 1)
                self.assertEqual(retry.json()["created_classes"], 0)
                self.assertEqual(self.annotations(image_id), labels)

    def test_new_classes_append_and_duplicate_files_share_an_image(self):
        original = self.client.post(
            f"/api/v1/projects/{self.project_id}/classes",
            headers=self.headers,
            json={"classes": [{"name": "scratch", "color": "#123456"}]},
        ).json()[0]
        grant = self.upload(
            {
                "scratch/a.png": picture(),
                "scratch/copy.png": picture(),
                "ok/b.png": picture(1),
            }
        )
        response = self.complete(grant)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["created_images"], 2)
        self.assertEqual(response.json()["duplicate_files"], 1)
        self.assertEqual(response.json()["created_classes"], 1)
        classes = sorted(self.rows(Class), key=lambda row: row.index)
        self.assertEqual(str(classes[0].id), original["id"])
        self.assertEqual(classes[0].color, "#123456")
        self.assertEqual((classes[1].index, classes[1].name), (1, "ok"))

    def test_import_labels_existing_unlabeled_image_without_reuploading_original(self):
        data = picture()
        with Session(self.engine) as session:
            image = Image(
                project_id=UUID(self.project_id),
                width=100,
                height=50,
                sha256=hashlib.sha256(data).hexdigest(),
                storage_key=f"test/{uuid4()}",
            )
            session.add(image)
            session.flush()
            image_id = str(image.id)
            self.store.put(image.storage_key, data, "image/png")
            session.commit()
        grant = self.upload()
        with patch.object(self.store, "put", wraps=self.store.put) as put:
            response = self.complete(grant)
            put.assert_not_called()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["labeled_images"], 1)
        self.assertEqual(response.json()["items"][0]["image_id"], image_id)
        self.assertEqual(len(self.annotations(image_id)["annotations"]), 1)

    def test_conflict_preserves_existing_labels_and_rolls_back_whole_batch(self):
        first = self.complete(self.upload()).json()
        image_id = first["items"][0]["image_id"]
        before = self.annotations(image_id)
        conflicting = self.upload({"other/a.png": picture(), "new/b.png": picture(1)})
        with patch.object(self.store, "put", wraps=self.store.put) as put:
            response = self.complete(conflicting)
            put.assert_not_called()
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.annotations(image_id), before)
        self.assertEqual(len(self.rows(Image)), 1)
        self.assertEqual([row.name for row in self.rows(Class)], ["scratch"])
        with Session(self.engine) as session:
            session.get(Image, UUID(image_id)).status = "in_review"
            session.commit()
        self.assertEqual(self.complete(self.upload()).status_code, 409)

    def test_validation_failure_leaves_no_database_rows_or_original_objects(self):
        cases = [
            {"ok/a.png": picture(), "bad/b.png": b"corrupt"},
            {"ok/a.png": picture(), "bad/b.png": picture()},
            {"../escape.png": picture()},
        ]
        for files in cases:
            grant = self.upload(files)
            with patch.object(self.store, "put", wraps=self.store.put) as put:
                response = self.complete(grant)
                put.assert_not_called()
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.rows(Image), [])
            self.assertEqual(self.rows(Class), [])

    def test_storage_failure_rolls_back_and_retry_reuses_deterministic_objects(self):
        grant = self.upload({"ok/a.png": picture(), "bad/b.png": picture(1)})
        original_put = self.store.put
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise EndpointConnectionError(endpoint_url="http://internal-only")
            return original_put(*args)

        with patch.object(self.store, "put", side_effect=fail_second):
            response = self.complete(grant)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json(), {"detail": "Object storage unavailable"})
        self.assertEqual(self.rows(Image), [])
        self.assertEqual(self.rows(Class), [])
        retry = self.complete(grant)
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(retry.json()["created_images"], 2)
        objects = self.store.client.list_objects_v2(
            Bucket=self.store.bucket, Prefix=f"{self.project_id}/originals/"
        )["Contents"]
        self.assertEqual(len(objects), 2)

    def test_missing_tampered_expired_and_wrong_scope_grants(self):
        data = zipped(dataset_files("imagefolder"))
        grant = self.permission(data)
        self.assertEqual(self.complete(grant).status_code, 409)
        _, key = read_import(
            grant["upload_token"], UUID(self.project_id), self.user_id, self.settings
        )
        self.store.put(key, b"x" * len(data), "application/zip")
        self.assertEqual(self.complete(grant).status_code, 422)
        secret = self.settings.jwt_secret.get_secret_value()
        claims = jwt.decode(
            grant["upload_token"],
            secret,
            algorithms=["HS256"],
            audience="visionforge-import",
        )
        for token in (
            grant["upload_token"] + "bad",
            self.headers["Authorization"][7:],
            jwt.encode(
                {**claims, "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
                secret,
                algorithm="HS256",
            ),
            jwt.encode(
                {**claims, "sub": str(self.other_id)}, secret, algorithm="HS256"
            ),
        ):
            self.assertEqual(self.complete({"upload_token": token}).status_code, 400)
        self.project_id = self.project("classification")
        self.assertEqual(self.complete(grant).status_code, 400)

    def test_ownership_task_matching_request_limits_and_openapi(self):
        body = claim_for(
            zipped(dataset_files("imagefolder")), "imagefolder"
        ).model_dump()
        grant = self.upload()
        for method, suffix, payload in (
            ("POST", "/presign", body),
            ("POST", "", {"upload_token": grant["upload_token"]}),
        ):
            for headers in ({}, self.auth(self.user_id, "refresh")):
                response = self.client.request(
                    method,
                    f"/api/v1/projects/{self.project_id}/imports{suffix}",
                    headers=headers,
                    json=payload,
                )
                self.assertEqual(response.status_code, 401)
            for project_id in (self.project_id, str(uuid4())):
                response = self.client.request(
                    method,
                    f"/api/v1/projects/{project_id}/imports{suffix}",
                    headers=self.auth(self.other_id),
                    json=payload,
                )
                self.assertEqual(response.status_code, 404)
        path = f"/api/v1/projects/{self.project_id}/imports/presign"
        for fields in (
            {"format": "yolo"},
            {"format": "unknown"},
            {"size": 64 * 1024 * 1024 + 1},
            {"root": "../bad"},
            {"annotations_file": "/bad.json"},
            {"key": "chosen-by-client"},
        ):
            self.assertEqual(
                self.client.post(
                    path, headers=self.headers, json={**body, **fields}
                ).status_code,
                422,
            )
        schema = self.client.get("/openapi.json").json()["paths"]
        for suffix in ("", "/presign"):
            self.assertEqual(
                schema["/api/v1/projects/{project_id}/imports" + suffix]["post"][
                    "security"
                ],
                [{"HTTPBearer": []}],
            )

    def test_busy_import_can_retry_and_queue_failure_does_not_undo_commit(self):
        grant = self.upload()
        self.assertTrue(slots.acquire(blocking=False))
        try:
            response = self.complete(grant)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["retry-after"], "5")
        finally:
            slots.release()
        with patch.object(self.worker, "send_task", side_effect=QueueError("offline")):
            response = self.complete(grant)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.rows(Image)), 1)
        with Session(self.engine) as session:
            self.assertIsNone(session.get(Image, self.rows(Image)[0].id).thumb_key)
        self.assertEqual(self.complete(grant).status_code, 200)

    def test_detection_background_imports_are_labeled_and_without_boxes(self):
        self.project_id = self.project("detection")
        files = dataset_files("yolo")
        files.pop("labels/a.txt")
        response = self.complete(self.upload(files, "yolo"))
        self.assertEqual(response.status_code, 200, response.text)
        result = self.annotations(response.json()["items"][0]["image_id"])
        self.assertEqual(result["status"], "labeled")
        self.assertEqual(result["annotations"], [])
        with Session(self.engine) as session:
            self.assertEqual(
                list(
                    session.scalars(
                        select(Annotation).where(
                            Annotation.image_id == UUID(result["image_id"])
                        )
                    )
                ),
                [],
            )

    def test_concurrent_label_edit_is_seen_after_waiting_for_image_lock(self):
        grant = self.upload()
        first = self.complete(grant).json()
        image_id = UUID(first["items"][0]["image_id"])
        self.client.put(
            f"/api/v1/images/{image_id}/annotations",
            headers=self.headers,
            json={"annotations": []},
        )
        other_class = self.client.post(
            f"/api/v1/projects/{self.project_id}/classes",
            headers=self.headers,
            json={"classes": [{"name": "other", "color": "#123456"}]},
        ).json()[0]
        locked = Event()

        def signal_project_lock(*args, **kwargs):
            result = owned_project(*args, **kwargs)
            locked.set()
            return result

        with Session(self.engine) as editor:
            image = editor.scalar(
                select(Image).where(Image.id == image_id).with_for_update()
            )
            with patch(
                "vf_api.dataset_imports.owned_project", side_effect=signal_project_lock
            ):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(self.complete, grant)
                    try:
                        self.assertTrue(locked.wait(timeout=10))
                        image.status = "labeled"
                        editor.add(
                            Annotation(
                                image_id=image_id,
                                class_id=UUID(other_class["id"]),
                                source="human",
                            )
                        )
                        editor.commit()
                    finally:
                        editor.rollback()
                    response = pending.result(timeout=10)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(
            self.annotations(image_id)["annotations"][0]["class_id"], other_class["id"]
        )
