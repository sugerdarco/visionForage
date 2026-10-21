"""SSE over a real HTTP socket with disposable PostgreSQL jobs and Redis channels."""

from datetime import datetime, timedelta, timezone
import json
import socket
from threading import BoundedSemaphore, Thread
import time
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx2 as httpx
from job_fixture import JobAPITest
import jwt
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
import uvicorn

from vf_api.job_events import JobPublisher, STAGES, channel
from vf_api.models import Model, Project, TrainingJob, TrainingMetric
from vf_api.settings import QueueSettings


def frames(response):
    fields = {}
    for line in response.iter_lines():
        if not line:
            if "data" in fields:
                fields["data"] = json.loads(fields["data"])
                yield fields
            fields = {}
        elif not line.startswith(":"):
            name, _, value = line.partition(":")
            fields[name] = value.lstrip()


class JobEventTest(JobAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        cls.addClassCleanup(listener.close)
        cls.url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        # APITest already owns the app lifespan; use the same database/session pool.
        cls.server = uvicorn.Server(
            uvicorn.Config(cls.app, lifespan="off", log_level="error", access_log=False)
        )
        cls.thread = Thread(
            target=cls.server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        cls.thread.start()
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 10
        while not cls.server.started:
            if time.monotonic() > deadline:
                raise AssertionError("HTTP test server did not start")
            time.sleep(0.01)
        cls.redis = Redis.from_url(QueueSettings().broker_url)
        cls.addClassCleanup(cls.redis.close)

    @classmethod
    def stop_server(cls):
        cls.server.should_exit = True
        cls.thread.join(10)
        if cls.thread.is_alive():
            raise AssertionError("HTTP test server did not stop")

    def setUp(self):
        super().setUp()
        self.identifier = UUID(self.create().json()["id"])
        self.endpoint = f"{self.url}/api/v1/jobs/{self.identifier}/events"
        self.publisher = JobPublisher()
        self.addCleanup(self.publisher.close)

    def state(self, status, *, model=False):
        with Session(self.engine) as session:
            job = session.get(TrainingJob, self.identifier)
            job.status = status
            if status != "queued":
                job.started_at = job.started_at or datetime.now(timezone.utc)
            if status in ("succeeded", "failed", "cancelled"):
                job.finished_at = datetime.now(timezone.utc)
            if model:
                row = Model(
                    project_id=self.project_id,
                    job_id=self.identifier,
                    version=1,
                    metrics={},
                    class_map={"ok": 0, "defect": 1},
                    weights_key="private/weights",
                    onnx_key="private/model",
                    input_size=224,
                )
                session.add(row)
                session.flush()
                model_id = str(row.id)
            session.commit()
        return model_id if model else None

    def metrics(self, *epochs):
        with Session(self.engine) as session:
            session.add_all(
                [
                    TrainingMetric(
                        job_id=self.identifier,
                        epoch=epoch,
                        split=split,
                        name=name,
                        value=value,
                    )
                    for epoch in epochs
                    for split, name, value in (
                        ("train", "loss", 0.3),
                        ("val", "accuracy", 0.8),
                        ("train", "learning_rate", 0.001),
                    )
                ]
            )
            session.commit()

    def test_live_epochs_logs_and_terminal_model_are_streamed_without_raw_broker_data(
        self,
    ):
        self.state("running")
        with httpx.stream(
            "GET", self.endpoint, headers=self.headers, timeout=8
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["x-accel-buffering"], "no")
            incoming = frames(response)
            self.assertEqual(next(incoming)["data"]["status"], "running")
            # Untrusted fields and raw log lines never get relayed.
            self.redis.publish(
                channel(self.identifier),
                json.dumps(
                    {
                        "type": "log",
                        "job_id": str(self.identifier),
                        "line": "private connection traceback",
                    }
                ),
            )
            self.publisher.log(self.identifier, "training")
            log = next(incoming)
            self.assertEqual(log["event"], "log")
            self.assertEqual(log["data"]["line"], STAGES["training"])
            self.metrics(1)
            self.publisher.publish(
                "epoch", self.identifier, epoch=99, metrics={"forged": 999}
            )
            epoch = next(incoming)
            self.assertEqual(epoch["id"], f"epoch:{self.identifier}:1")
            self.assertEqual(
                epoch["data"]["metrics"],
                {"train_loss": 0.3, "val_accuracy": 0.8, "lr": 0.001},
            )
            model_id = self.state("succeeded", model=True)
            self.publisher.publish(
                "done", self.identifier, status="failed", model_id="forged"
            )
            remaining = list(incoming)
            self.assertEqual([item["event"] for item in remaining], ["status", "done"])
            self.assertEqual(remaining[-1]["data"]["model_id"], model_id)
            self.assertEqual(remaining[-1]["data"]["status"], "succeeded")
            for frame in [log, epoch, *remaining]:
                self.assertEqual(frame["data"]["job_id"], str(self.identifier))
                self.assertIsNotNone(datetime.fromisoformat(frame["data"]["ts"]).tzinfo)

    def test_reconnect_replays_complete_epochs_in_batches_before_done(self):
        self.metrics(*range(1, 46))
        self.state("failed")
        headers = {**self.headers, "Last-Event-ID": f"epoch:{self.identifier}:2"}
        with httpx.stream("GET", self.endpoint, headers=headers, timeout=8) as response:
            result = list(frames(response))
        self.assertEqual(result[0]["event"], "status")
        epochs = [item["data"]["epoch"] for item in result if item["event"] == "epoch"]
        self.assertEqual(epochs, list(range(3, 46)))
        self.assertEqual(result[-1]["event"], "done")
        headers["Last-Event-ID"] = f"epoch:{self.identifier}:45"
        with httpx.stream("GET", self.endpoint, headers=headers, timeout=8) as response:
            self.assertEqual(
                [item["event"] for item in frames(response)], ["status", "done"]
            )

    def test_owner_access_token_cursor_validation_and_cors(self):
        for headers, expected in (
            ({}, 401),
            (self.auth(self.user_id, "refresh"), 401),
            (self.auth(self.other_id), 404),
        ):
            response = httpx.get(self.endpoint, headers=headers, timeout=5)
            self.assertEqual(response.status_code, expected, response.text)
        self.assertEqual(
            httpx.get(
                self.endpoint.replace(str(self.identifier), str(uuid4())),
                headers=self.headers,
            ).status_code,
            404,
        )
        for cursor in (
            "bad",
            f"epoch:{uuid4()}:1",
            f"epoch:{self.identifier}:-1",
            f"epoch:{self.identifier}:1001",
            f"epoch:{self.identifier}:1",
            "x" * 81,
        ):
            response = httpx.get(
                self.endpoint,
                headers={**self.headers, "Last-Event-ID": cursor},
                timeout=5,
            )
            self.assertEqual(response.status_code, 422, response.text)
        response = httpx.options(
            self.endpoint,
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization,Last-Event-ID",
            },
        )
        self.assertEqual(response.status_code, 200)
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/v1/jobs/{job_id}/events"
        ]["get"]
        self.assertEqual(operation["security"], [{"HTTPBearer": []}])
        self.assertIn("text/event-stream", operation["responses"]["200"]["content"])

    def test_redis_outage_falls_back_to_database_and_recovery_status(self):
        with patch("vf_api.job_stream.Redis.from_url") as factory:
            from unittest.mock import AsyncMock

            client = factory.return_value
            client.aclose = AsyncMock()
            subscription = client.pubsub.return_value
            subscription.subscribe = AsyncMock(
                side_effect=RedisConnectionError("private broker failure")
            )
            subscription.aclose = AsyncMock()
            with httpx.stream(
                "GET", self.endpoint, headers=self.headers, timeout=8
            ) as response:
                incoming = frames(response)
                self.assertEqual(next(incoming)["data"]["status"], "queued")
                self.metrics(1)
                self.state("failed")
                result = list(incoming)
                self.assertEqual(
                    [item["event"] for item in result], ["status", "epoch", "done"]
                )
            self.assertTrue(client.aclose.called)

    def test_stream_rechecks_expiring_authentication_and_changed_ownership(self):
        claims = jwt.decode(
            self.headers["Authorization"].split()[1],
            options={"verify_signature": False},
        )
        claims["exp"] = datetime.now(timezone.utc) + timedelta(seconds=3)
        token = jwt.encode(
            claims, self.settings.jwt_secret.get_secret_value(), algorithm="HS256"
        )
        with httpx.stream(
            "GET",
            self.endpoint,
            headers={"Authorization": "Bearer " + token},
            timeout=8,
        ) as response:
            result = list(frames(response))
            self.assertEqual(result[-1]["data"]["reason"], "authentication_required")
        with httpx.stream(
            "GET", self.endpoint, headers=self.headers, timeout=8
        ) as response:
            incoming = frames(response)
            next(incoming)
            with Session(self.engine) as session:
                session.get(Project, self.project_id).owner_id = self.other_id
                session.commit()
            self.publisher.publish("status", self.identifier, status="running")
            self.assertEqual(next(incoming)["data"]["reason"], "job_not_found")
            self.assertEqual(list(incoming), [])

    def test_disconnect_releases_subscription_slot_and_database_connection(self):
        slots = BoundedSemaphore(1)
        old = self.app.state.job_stream_slots
        self.app.state.job_stream_slots = slots
        self.addCleanup(setattr, self.app.state, "job_stream_slots", old)
        with httpx.stream(
            "GET", self.endpoint, headers=self.headers, timeout=8
        ) as response:
            incoming = frames(response)
            next(incoming)
            self.assertEqual(
                httpx.get(self.endpoint, headers=self.headers, timeout=5).status_code,
                429,
            )
            self.assertEqual(self.app.state.sessions.kw["bind"].pool.checkedout(), 0)
            self.assertEqual(
                self.redis.pubsub_numsub(channel(self.identifier))[0][1], 1
            )
        deadline = time.monotonic() + 5
        while self.redis.pubsub_numsub(channel(self.identifier))[0][
            1
        ] or not slots.acquire(blocking=False):
            if time.monotonic() > deadline:
                self.fail("Stream resources were not released after disconnect")
            time.sleep(0.02)
        slots.release()

    def test_keepalive_rotation_and_sanitized_database_failure(self):
        with (
            patch("vf_api.job_stream.KEEPALIVE_SECONDS", 0),
            patch("vf_api.job_stream.STREAM_SECONDS", 0.1),
        ):
            response = httpx.get(self.endpoint, headers=self.headers, timeout=8)
            self.assertIn(": keep-alive", response.text)
            self.assertIn('"type": "reconnect"', response.text)
        with httpx.stream(
            "GET", self.endpoint, headers=self.headers, timeout=8
        ) as response:
            incoming = frames(response)
            next(incoming)
            with patch(
                "vf_api.job_stream.snapshot",
                side_effect=OperationalError(
                    "private sql", {}, Exception("private address")
                ),
            ):
                self.publisher.publish("status", self.identifier, status="running")
                result = next(incoming)
                self.assertEqual(result["data"]["reason"], "database_unavailable")
                self.assertNotIn("private", json.dumps(result))
