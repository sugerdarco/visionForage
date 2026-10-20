"""Real S3 PUTs and Redis worker delivery, isolated from application data."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import time
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

from api_fixture import APITest
from botocore.exceptions import EndpointConnectionError
from celery.contrib.testing.worker import start_worker
import httpx2 as httpx
import jwt
from kombu.exceptions import OperationalError as QueueError
from PIL import Image as PillowImage
from redis import Redis
import sqlalchemy as sa
from sqlalchemy.orm import Session

from vf_api.models import Annotation, Class, Image, User
from vf_api.queueing import RECOVERY_TASK, THUMBNAIL_TASK
from vf_api.security import issue_token
from vf_api.settings import QueueSettings, StorageSettings
from vf_api.storage import MAX_UPLOAD_BYTES, ObjectStore
from vf_api.upload_dependencies import get_queue, get_store
from vf_api.upload_tokens import read_upload
from vf_worker.app import create_worker


def image_bytes(mode="RGB", size=(640, 480), format="PNG", orientation=None):
    output = BytesIO()
    with PillowImage.new(mode, size, color=0) as image:
        options = {}
        if orientation is not None:
            exif = image.getexif()
            exif[274] = orientation
            options["exif"] = exif
        image.save(output, format=format, **options)
    return output.getvalue()


class UploadTest(APITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        suffix = uuid4().hex
        cls.storage_settings = StorageSettings(s3_images_bucket="vf-test-" + suffix)
        cls.store = ObjectStore(cls.storage_settings)
        cls.addClassCleanup(cls.store.close)
        cls.store.client.create_bucket(Bucket=cls.store.bucket)
        cls.addClassCleanup(cls.drop_bucket)
        cls.store.configure_staging_expiry()
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
        cls.consumer = start_worker(
            cls.worker,
            perform_ping_check=False,
            pool="solo",
            concurrency=1,
            loglevel="ERROR",
            shutdown_timeout=15,
        )
        cls.consumer.__enter__()
        cls.addClassCleanup(cls.consumer.__exit__, None, None, None)

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
            session.add(user)
            session.flush()
            self.user_id = user.id
            session.commit()
        token, _ = issue_token(self.user_id, "access", self.settings)
        self.headers = {"Authorization": f"Bearer {token}"}
        response = self.client.post(
            "/api/v1/projects",
            headers=self.headers,
            json={"name": "Upload test", "task_type": "detection"},
        )
        self.assertEqual(response.status_code, 201)
        self.project_id = response.json()["id"]
        self.path = f"/api/v1/projects/{self.project_id}/uploads"

    def permission(self, data, mime="image/png"):
        response = self.client.post(
            self.path + "/presign",
            headers=self.headers,
            json={
                "files": [
                    {
                        "content_type": mime,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()[0]

    def upload(self, data=None, mime="image/png"):
        data = data if data is not None else image_bytes()
        grant = self.permission(data, mime)
        response = httpx.put(
            grant["upload_url"], content=data, headers=grant["headers"], timeout=15
        )
        self.assertEqual(response.status_code, 200, response.text)
        return grant

    def complete(self, grant):
        return self.client.post(
            self.path + "/complete",
            headers=self.headers,
            json={"upload_token": grant["upload_token"]},
        )

    def wait_thumbnail(self, image_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = self.client.get(
                f"/api/v1/images/{image_id}", headers=self.headers
            )
            self.assertEqual(response.status_code, 200, response.text)
            if response.json()["thumbnail_url"]:
                return response.json()
            time.sleep(0.1)
        self.fail("Redis worker did not finish the thumbnail within 15 seconds")

    def test_direct_upload_registration_worker_and_private_reads(self):
        data = image_bytes()
        grant = self.upload(data)
        signed = parse_qs(urlsplit(grant["upload_url"]).query)["X-Amz-SignedHeaders"][0]
        for header in ("content-length", "content-type", "x-amz-checksum-sha256"):
            self.assertIn(header, signed.split(";"))
        response = self.complete(grant)
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()
        self.assertEqual((record["width"], record["height"]), (640, 480))
        self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(record["status"], "unlabeled")
        self.assertEqual(response.headers["cache-control"], "no-store")
        preview = self.wait_thumbnail(record["id"])
        original = httpx.get(preview["image_url"], timeout=15)
        self.assertEqual(original.content, data)
        thumbnail = httpx.get(preview["thumbnail_url"], timeout=15)
        self.assertEqual(thumbnail.status_code, 200)
        with PillowImage.open(BytesIO(thumbnail.content)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (320, 240))
        unsigned = preview["image_url"].split("?", 1)[0]
        self.assertEqual(httpx.get(unsigned, timeout=15).status_code, 403)
        with Session(self.engine) as session:
            row = session.get(Image, UUID(record["id"]))
            self.assertNotIn("uploads/", row.storage_key)
        # Replaying the staging PUT cannot overwrite the registered original.
        wrong = httpx.put(
            grant["upload_url"],
            content=b"x" * len(data),
            headers=grant["headers"],
            timeout=15,
        )
        self.assertGreaterEqual(wrong.status_code, 400)
        self.assertEqual(httpx.get(preview["image_url"], timeout=15).content, data)

    def test_repeated_and_concurrent_completion_deduplicates_per_project(self):
        grant = self.upload(image_bytes(mode="L"))
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: self.complete(grant), range(2)))
        self.assertEqual(
            sorted(response.status_code for response in responses), [200, 201]
        )
        self.assertEqual(responses[0].json()["id"], responses[1].json()["id"])
        repeated = self.complete(grant)
        self.assertEqual(repeated.status_code, 200)
        second = self.upload(image_bytes(mode="L"))
        self.assertEqual(self.complete(second).json()["id"], repeated.json()["id"])
        listing = self.client.get(
            f"/api/v1/projects/{self.project_id}/images", headers=self.headers
        )
        self.assertEqual(listing.json()["total"], 1)
        self.wait_thumbnail(repeated.json()["id"])
        other = self.client.post(
            "/api/v1/projects",
            headers=self.headers,
            json={"name": "Separate dataset", "task_type": "classification"},
        ).json()
        self.path = f"/api/v1/projects/{other['id']}/uploads"
        separate = self.complete(self.upload(image_bytes(mode="L")))
        self.assertEqual(separate.status_code, 201, separate.text)
        self.assertNotEqual(separate.json()["id"], repeated.json()["id"])

    def test_gallery_filters_and_pagination(self):
        first = self.complete(self.upload()).json()
        second = self.complete(
            self.upload(image_bytes(format="JPEG"), "image/jpeg")
        ).json()
        with Session(self.engine) as session:
            label = Class(
                project_id=UUID(self.project_id),
                name="scratch",
                color="#112233",
                index=0,
            )
            session.add(label)
            session.flush()
            label_id = str(label.id)
            session.get(Image, UUID(first["id"])).status = "labeled"
            session.add_all(
                [
                    Annotation(
                        image_id=UUID(first["id"]),
                        class_id=label.id,
                        source="human",
                        bbox=[0, 0, 0.2, 0.2],
                    ),
                    Annotation(
                        image_id=UUID(first["id"]),
                        class_id=label.id,
                        source="human",
                        bbox=[0.5, 0.5, 0.2, 0.2],
                    ),
                ]
            )
            session.commit()
        path = f"/api/v1/projects/{self.project_id}/images"
        listing = self.client.get(path + "?page_size=1", headers=self.headers).json()
        self.assertEqual(listing["total"], 2)
        self.assertEqual(listing["items"][0]["id"], second["id"])
        page = self.client.get(
            path + "?page=2&page_size=1", headers=self.headers
        ).json()
        self.assertEqual(page["items"][0]["id"], first["id"])
        filtered = self.client.get(
            path + f"?status=labeled&class={label_id}", headers=self.headers
        ).json()
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["id"], first["id"])
        self.assertEqual(
            self.client.get(path + f"?class={uuid4()}", headers=self.headers).json()[
                "total"
            ],
            0,
        )
        for query in ("page=0", "page_size=101", "status=unknown", "class=invalid"):
            self.assertEqual(
                self.client.get(path + "?" + query, headers=self.headers).status_code,
                422,
            )

    def test_storage_failure_is_generic_and_registration_can_retry(self):
        grant = self.upload()
        with patch.object(
            self.store,
            "read",
            side_effect=EndpointConnectionError(
                endpoint_url="http://private-internal-host"
            ),
        ):
            response = self.complete(grant)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Object storage unavailable"})
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.complete(grant).status_code, 201)

    def test_invalid_images_are_not_registered(self):
        cases = [
            (b"not an image", "image/png"),
            (image_bytes(mode="RGBA"), "image/png"),
            (image_bytes(size=(9, 10)), "image/png"),
            (image_bytes(format="JPEG", orientation=6), "image/jpeg"),
            (image_bytes(format="JPEG"), "image/png"),
            (image_bytes()[:80], "image/png"),
            (image_bytes(mode="L", size=(5000, 5000)), "image/png"),
        ]
        for data, mime in cases:
            with self.subTest(size=len(data), mime=mime):
                response = self.complete(self.upload(data, mime))
                self.assertEqual(response.status_code, 422, response.text)
        with Session(self.engine) as session:
            count = session.scalar(
                sa.select(sa.func.count())
                .select_from(Image)
                .where(Image.project_id == UUID(self.project_id))
            )
            self.assertEqual(count, 0)

    def test_missing_or_corrupt_staging_object_and_bad_claims(self):
        data = image_bytes()
        grant = self.permission(data)
        self.assertEqual(self.complete(grant).status_code, 409)
        _, key = read_upload(
            grant["upload_token"], UUID(self.project_id), self.user_id, self.settings
        )
        self.store.put(key, b"corrupted", "image/png")
        self.assertEqual(self.complete(grant).status_code, 422)
        for file in (
            {"content_type": "image/gif", "size": 10, "sha256": "a" * 64},
            {
                "content_type": "image/png",
                "size": MAX_UPLOAD_BYTES + 1,
                "sha256": "a" * 64,
            },
            {"content_type": "image/png", "size": 0, "sha256": "a" * 64},
            {"content_type": "image/png", "size": 10, "sha256": "../escape"},
            {
                "content_type": "image/png",
                "size": 10,
                "sha256": "a" * 64,
                "key": "other/file",
            },
        ):
            response = self.client.post(
                self.path + "/presign", headers=self.headers, json={"files": [file]}
            )
            self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self.client.post(
                self.path + "/presign", headers=self.headers, json={"files": []}
            ).status_code,
            422,
        )

    def test_grant_expiry_tampering_and_project_scope(self):
        grant = self.upload()
        secret = self.settings.jwt_secret.get_secret_value()
        claims = jwt.decode(
            grant["upload_token"],
            secret,
            algorithms=["HS256"],
            audience="visionforge-upload",
        )
        expired = jwt.encode(
            {**claims, "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
            secret,
            algorithm="HS256",
        )
        for token in (
            expired,
            grant["upload_token"] + "tampered",
            self.headers["Authorization"][7:],
        ):
            self.assertEqual(self.complete({"upload_token": token}).status_code, 400)
        other = self.client.post(
            "/api/v1/projects",
            headers=self.headers,
            json={"name": "Other", "task_type": "classification"},
        ).json()
        response = self.client.post(
            f"/api/v1/projects/{other['id']}/uploads/complete",
            headers=self.headers,
            json={"upload_token": grant["upload_token"]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.complete(grant).status_code, 201)

    def test_other_owner_cannot_presign_complete_list_or_read(self):
        grant = self.upload()
        record = self.complete(grant).json()
        with Session(self.engine) as session:
            other = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            session.add(other)
            session.flush()
            token, _ = issue_token(other.id, "access", self.settings)
            session.commit()
        headers = {"Authorization": f"Bearer {token}"}
        file = {"content_type": "image/png", "size": 10, "sha256": "a" * 64}
        for method, path, body in (
            ("POST", self.path + "/presign", {"files": [file]}),
            ("POST", self.path + "/complete", {"upload_token": grant["upload_token"]}),
            ("GET", f"/api/v1/projects/{self.project_id}/images", None),
            ("GET", f"/api/v1/images/{record['id']}", None),
        ):
            self.assertEqual(
                self.client.request(
                    method, path, headers=headers, json=body
                ).status_code,
                404,
            )
            self.assertEqual(
                self.client.request(method, path, json=body).status_code, 401
            )

    def test_queue_outage_is_recovered_from_pending_rows(self):
        grant = self.upload(image_bytes(format="JPEG"), "image/jpeg")
        with patch.object(self.worker, "send_task", side_effect=QueueError("offline")):
            response = self.complete(grant)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertIsNone(response.json()["thumbnail_url"])
        self.worker.send_task(RECOVERY_TASK)
        recovered = self.wait_thumbnail(response.json()["id"])
        before = httpx.get(recovered["thumbnail_url"], timeout=15).content
        self.worker.send_task(THUMBNAIL_TASK, args=[response.json()["id"]])
        # A synchronous redelivery also proves the task's completed-row fast path.
        self.worker.tasks[THUMBNAIL_TASK].apply(
            args=[response.json()["id"]], throw=True
        )
        self.assertEqual(
            httpx.get(recovered["thumbnail_url"], timeout=15).content, before
        )

    def test_lifecycle_and_browser_cors(self):
        api_preflight = self.client.options(
            self.path + "/presign",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        self.assertEqual(api_preflight.status_code, 200)
        self.assertEqual(
            api_preflight.headers["access-control-allow-origin"],
            "http://localhost:3000",
        )
        rules = self.store.client.get_bucket_lifecycle_configuration(
            Bucket=self.store.bucket
        )["Rules"]
        self.assertEqual(rules[0]["Filter"], {"Prefix": "uploads/"})
        self.assertEqual(rules[0]["Expiration"], {"Days": 1})
        self.store.configure_staging_expiry()
        self.assertEqual(
            len(
                self.store.client.get_bucket_lifecycle_configuration(
                    Bucket=self.store.bucket
                )["Rules"]
            ),
            1,
        )
        grant = self.permission(image_bytes())
        response = httpx.options(
            grant["upload_url"],
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type,x-amz-checksum-sha256",
            },
            timeout=15,
        )
        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(
            response.headers["access-control-allow-origin"], "http://localhost:3000"
        )
