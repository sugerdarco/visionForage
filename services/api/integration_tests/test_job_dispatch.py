"""Durable training dispatch against disposable PostgreSQL and Redis queues."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from threading import Event
import time
from unittest.mock import patch
from uuid import UUID, uuid4

from celery.contrib.testing.worker import start_worker
from job_fixture import JobAPITest
from kombu.exceptions import OperationalError as QueueError
from redis import Redis
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from vf_api.database import make_session_factory
from vf_api.models import TrainingJob
from vf_api.queueing import DISPATCH_TASK, TRAIN_TASK
from vf_api.settings import QueueSettings
from vf_worker.app import create_worker
from vf_worker.dispatch import dispatch_pending


class JobDispatchTest(JobAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.queue_settings = QueueSettings()
        cls.worker = create_worker(database=cls.settings, queue=cls.queue_settings)
        cls.addClassCleanup(cls.worker.close)
        cls.prefix = "vf-test-" + uuid4().hex + ":"
        cls.cpu_queue, cls.gpu_queue = cls.prefix + "cpu", cls.prefix + "gpu"
        cls.worker.conf.task_default_queue = cls.cpu_queue
        cls.worker.conf.task_routes = {TRAIN_TASK: {"queue": cls.gpu_queue}}
        cls.worker.conf.broker_transport_options = {
            **cls.worker.conf.broker_transport_options,
            "global_keyprefix": cls.prefix,
        }
        cls.addClassCleanup(cls.drop_queue)
        cls.sessions = make_session_factory(cls.engine)

    @classmethod
    def drop_queue(cls):
        with Redis.from_url(cls.queue_settings.broker_url) as client:
            keys = list(client.scan_iter(match=cls.prefix + "*"))
            if keys:
                client.delete(*keys)

    def tearDown(self):
        # Each method leaves its rows for inspection but no work for the next method.
        with Session(self.engine) as session:
            session.execute(
                update(TrainingJob)
                .where(TrainingJob.project_id == self.project_id)
                .values(status="cancelled")
            )
            session.commit()
        with self.worker.connection_for_read() as connection:
            with connection.SimpleQueue(self.gpu_queue) as queue:
                queue.clear()

    def dispatch(self, **kwargs):
        return dispatch_pending(self.sessions, self.worker, **kwargs)

    def messages(self):
        found = []
        with self.worker.connection_for_read() as connection:
            with connection.SimpleQueue(self.gpu_queue) as queue:
                while True:
                    try:
                        message = queue.get(block=False)
                    except Empty:
                        break
                    found.append(
                        {"headers": message.headers, "payload": message.payload}
                    )
                    message.ack()
        return found

    def test_cpu_dispatch_task_sends_only_job_id_to_separate_gpu_queue(self):
        response = self.create()
        self.assertEqual(response.status_code, 201, response.text)
        identifier = response.json()["id"]
        with start_worker(
            self.worker,
            queues=[self.cpu_queue],
            perform_ping_check=False,
            pool="solo",
            concurrency=1,
            loglevel="ERROR",
            shutdown_timeout=15,
        ):
            self.worker.send_task(DISPATCH_TASK)
            deadline = time.monotonic() + 10
            while self.read(identifier).json()["dispatched_at"] is None:
                if time.monotonic() >= deadline:
                    self.fail("CPU dispatcher did not publish the training request")
                time.sleep(0.05)
        messages = self.messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["headers"]["task"], TRAIN_TASK)
        self.assertEqual(messages[0]["payload"][:2], [[identifier], {}])
        with Session(self.engine) as session:
            row = session.get(TrainingJob, UUID(identifier))
            self.assertEqual(messages[0]["headers"]["id"], row.celery_task_id)
            self.assertEqual(row.status, "queued")
            self.assertIsNone(row.started_at)
        self.assertEqual(self.dispatch(), 0)
        self.assertEqual(self.messages(), [])

    def test_redis_failure_leaves_job_pending_and_later_dispatch_recovers(self):
        identifier = self.create().json()["id"]
        with patch.object(
            self.worker, "send_task", side_effect=QueueError("unavailable")
        ):
            with self.assertRaises(QueueError):
                self.dispatch()
        row = self.read(identifier).json()
        self.assertEqual(row["status"], "queued")
        self.assertIsNone(row["dispatched_at"])
        self.assertEqual(self.dispatch(), 1)
        self.assertEqual(len(self.messages()), 1)
        self.assertIsNotNone(self.read(identifier).json()["dispatched_at"])

    def test_database_failure_after_send_reuses_task_id_when_republished(self):
        identifier = self.create().json()["id"]
        # The broker may accept a message just before the database commit fails.
        def fail_update(connection, cursor, statement, parameters, context, many):
            if statement.startswith("UPDATE training_jobs SET dispatched_at"):
                raise OperationalError("UPDATE", {}, Exception("unavailable"))

        event.listen(self.engine, "before_cursor_execute", fail_update)
        try:
            with self.assertRaises(OperationalError):
                self.dispatch()
        finally:
            event.remove(self.engine, "before_cursor_execute", fail_update)
        first = self.messages()
        self.assertEqual(len(first), 1)
        self.assertIsNone(self.read(identifier).json()["dispatched_at"])
        self.assertEqual(self.dispatch(), 1)
        second = self.messages()
        self.assertEqual(first, second)

    def test_concurrent_dispatchers_do_not_publish_the_same_locked_job(self):
        self.create()
        sending, resume = Event(), Event()
        original = self.worker.send_task

        def pause(*args, **kwargs):
            sending.set()
            if not resume.wait(10):
                raise AssertionError("Publisher not resumed")
            return original(*args, **kwargs)

        with patch.object(self.worker, "send_task", side_effect=pause):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.dispatch)
                try:
                    self.assertTrue(sending.wait(10))
                    self.assertEqual(pool.submit(self.dispatch).result(5), 0)
                finally:
                    resume.set()
                self.assertEqual(first.result(10), 1)
        self.assertEqual(len(self.messages()), 1)

    def test_cancellation_before_and_after_publish_keeps_terminal_state(self):
        first = self.create().json()["id"]
        self.cancel(first)
        self.assertEqual(self.dispatch(), 0)
        self.assertEqual(self.messages(), [])
        second = self.create().json()["id"]
        self.assertEqual(self.dispatch(), 1)
        self.assertEqual(self.cancel(second).json()["status"], "cancelled")
        self.assertEqual(len(self.messages()), 1)
        # Queue messages aren't deleted; the consumer must check the authoritative row.
        self.assertEqual(self.dispatch(), 0)

    def test_cancellation_serializes_with_inflight_dispatch(self):
        identifier = self.create().json()["id"]
        sending, resume, cancelling = Event(), Event(), Event()
        original = self.worker.send_task

        def pause(*args, **kwargs):
            sending.set()
            if not resume.wait(10):
                raise AssertionError("Publisher not resumed")
            return original(*args, **kwargs)

        def cancel():
            cancelling.set()
            return self.cancel(identifier)

        with patch.object(self.worker, "send_task", side_effect=pause):
            with ThreadPoolExecutor(max_workers=2) as pool:
                dispatch = pool.submit(self.dispatch)
                try:
                    self.assertTrue(sending.wait(10))
                    cancellation = pool.submit(cancel)
                    self.assertTrue(cancelling.wait(5))
                finally:
                    resume.set()
                self.assertEqual(dispatch.result(10), 1)
                self.assertEqual(cancellation.result(10).json()["status"], "cancelled")
        self.assertEqual(self.dispatch(), 0)
        self.assertEqual(len(self.messages()), 1)

    def test_batch_limit_leaves_remaining_jobs_for_next_dispatch(self):
        identifiers = [self.create().json()["id"] for _ in range(3)]
        self.assertEqual(self.dispatch(limit=2), 2)
        self.assertEqual(self.dispatch(limit=2), 1)
        self.assertEqual(self.dispatch(limit=2), 0)
        self.assertEqual(
            [message["payload"][0][0] for message in self.messages()], identifiers
        )
        with Session(self.engine) as session:
            self.assertTrue(
                all(
                    row.dispatched_at
                    for row in session.scalars(
                        select(TrainingJob).where(
                            TrainingJob.project_id == self.project_id
                        )
                    )
                )
            )
