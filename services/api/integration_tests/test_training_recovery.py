"""Worker leases and fencing against a disposable PostgreSQL database."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch
from uuid import UUID, uuid4

from job_fixture import JobAPITest
from sqlalchemy import func, select, update

from vf_api.database import make_session_factory
from vf_api.models import Model, TrainingJob, TrainingMetric
from vf_worker.leases import EXPIRED_ERROR, Heartbeat, LeaseLost, recover_stalled, renew
from vf_worker.training import (
    Cancelled,
    check_active,
    claim,
    finish_unsuccessful,
    record_history,
    register,
)


class TrainingRecoveryTest(JobAPITest):
    def setUp(self):
        super().setUp()
        self.sessions = make_session_factory(self.engine)

    def tearDown(self):
        with self.sessions.begin() as session:
            session.execute(
                update(TrainingJob)
                .where(TrainingJob.project_id == self.project_id)
                .values(status="cancelled")
            )

    def running(self):
        identifier = UUID(self.create().json()["id"])
        with self.sessions() as session:
            task_id = session.get(TrainingJob, identifier).celery_task_id
        job, _, _ = claim(self.sessions, identifier, task_id, 1)
        return job

    def age(self, job):
        with self.sessions.begin() as session:
            session.get(TrainingJob, job.id).heartbeat_at = session.scalar(
                select(func.clock_timestamp() - timedelta(minutes=4))
            )

    def test_heartbeat_renews_only_live_uncancelled_owner(self):
        job = self.running()
        self.assertFalse(renew(self.sessions, job.id, uuid4()))
        self.assertTrue(renew(self.sessions, job.id, job.claim_token))
        with self.sessions() as session:
            self.assertGreater(
                session.get(TrainingJob, job.id).heartbeat_at, job.heartbeat_at
            )
        self.age(job)
        self.assertFalse(renew(self.sessions, job.id, job.claim_token))
        with self.assertRaises(Cancelled):
            check_active(self.sessions, job.id, job.claim_token)
        other = self.running()
        self.cancel(other.id)
        self.assertFalse(renew(self.sessions, other.id, other.claim_token))

    def test_recovery_marks_expired_jobs_failed_without_claiming_gpu_has_stopped(self):
        expired, cancelling, healthy = self.running(), self.running(), self.running()
        queued = self.create().json()["id"]
        self.age(expired)
        self.age(cancelling)
        self.cancel(cancelling.id)
        self.assertEqual(recover_stalled(self.sessions), 2)
        for job in (expired, cancelling):
            result = self.read(job.id).json()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"], EXPIRED_ERROR)
            self.assertIsNotNone(result["finished_at"])
            self.assertIsNone(result["model_id"])
            finish_unsuccessful(self.sessions, job.id, job.claim_token, "late failure")
            self.assertEqual(self.read(job.id).json(), result)
        self.assertEqual(self.read(healthy.id).json()["status"], "running")
        self.assertEqual(self.read(queued).json()["status"], "queued")
        self.assertEqual(recover_stalled(self.sessions), 0)

    def test_recovery_skips_locked_rows_and_bounds_each_pass(self):
        jobs = [self.running() for _ in range(3)]
        for job in jobs:
            self.age(job)
        with self.sessions.begin() as session:
            session.execute(
                select(TrainingJob)
                .where(TrainingJob.id == jobs[0].id)
                .with_for_update()
            ).one()
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(
                    pool.submit(recover_stalled, self.sessions).result(5), 2
                )
            self.assertEqual(session.get(TrainingJob, jobs[0].id).status, "running")
        self.assertEqual(recover_stalled(self.sessions, limit=1), 1)
        self.assertEqual(recover_stalled(self.sessions), 0)
        for _ in range(2):
            self.age(self.running())
        self.assertEqual(recover_stalled(self.sessions, limit=1), 1)
        self.assertEqual(recover_stalled(self.sessions, limit=1), 1)

    def test_replaced_claim_blocks_old_metrics_registration_and_failure(self):
        job = self.running()
        replacement = uuid4()
        with self.sessions.begin() as session:
            session.get(TrainingJob, job.id).claim_token = replacement
        with self.assertRaises(Cancelled):
            check_active(self.sessions, job.id, job.claim_token)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(
                '[{"epoch":1,"fitness":0.5,"metrics":{"train/box_loss":0.7}}]'
            )
            with self.assertRaises(Cancelled):
                record_history(self.sessions, job.id, job.claim_token, path, 0)
        with self.assertRaises(Cancelled):
            register(
                self.sessions, job.id, job.claim_token, job.project_id, {}, {}, {}, 224
            )
        finish_unsuccessful(self.sessions, job.id, job.claim_token, "old owner")
        self.assertFalse(renew(self.sessions, job.id, job.claim_token))
        with self.sessions() as session:
            self.assertEqual(session.get(TrainingJob, job.id).status, "running")
            self.assertIsNone(
                session.scalar(
                    select(TrainingMetric.id).where(TrainingMetric.job_id == job.id)
                )
            )
            self.assertIsNone(
                session.scalar(select(Model.id).where(Model.job_id == job.id))
            )
        check_active(self.sessions, job.id, replacement)

    def test_expired_lease_cannot_register_even_before_recovery_runs(self):
        job = self.running()
        self.age(job)
        with self.assertRaises(Cancelled):
            register(
                self.sessions, job.id, job.claim_token, job.project_id, {}, {}, {}, 224
            )
        self.assertEqual(recover_stalled(self.sessions), 1)

    def test_background_heartbeat_runs_without_progress_callbacks_and_stops(self):
        job = self.running()
        pulsed = Event()

        def observed(*args):
            result = renew(*args)
            pulsed.set()
            return result

        with (
            patch("vf_worker.leases.HEARTBEAT_SECONDS", 0.02),
            patch("vf_worker.leases.renew", side_effect=observed),
        ):
            with Heartbeat(self.sessions, job.id, job.claim_token) as heartbeat:
                self.assertTrue(pulsed.wait(5))
                heartbeat.check()
                self.assertEqual(recover_stalled(self.sessions), 0)
            self.assertFalse(heartbeat.thread.is_alive())
        with self.sessions() as session:
            self.assertGreater(
                session.get(TrainingJob, job.id).heartbeat_at, job.heartbeat_at
            )

    def test_heartbeat_failure_and_stuck_thread_signal_loss(self):
        job = self.running()
        with (
            patch("vf_worker.leases.HEARTBEAT_SECONDS", 0.02),
            patch(
                "vf_worker.leases.renew",
                side_effect=RuntimeError("private connection error"),
            ),
        ):
            with Heartbeat(self.sessions, job.id, job.claim_token) as heartbeat:
                self.assertTrue(heartbeat.lost.wait(5))
                with self.assertRaises(LeaseLost):
                    heartbeat.check()
        heartbeat = Heartbeat(self.sessions, job.id, job.claim_token)
        heartbeat.last_success -= 91
        with self.assertRaises(LeaseLost):
            heartbeat.check()
