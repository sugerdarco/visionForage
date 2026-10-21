"""Database-clock leases fence a worker that has lost ownership of its job."""

from datetime import timedelta
from threading import Event, Thread
import time

from sqlalchemy import func, select, update

from vf_api.models import TrainingJob

HEARTBEAT_SECONDS = 10
LEASE_SECONDS = 180
EXPIRED_ERROR = "Worker heartbeat expired; execution state unknown"


class LeaseLost(Exception):
    pass


def owned(identifier, token):
    return (
        TrainingJob.id == identifier,
        TrainingJob.status == "running",
        TrainingJob.claim_token == token,
        TrainingJob.cancel_requested_at.is_(None),
        TrainingJob.heartbeat_at
        >= func.clock_timestamp() - timedelta(seconds=LEASE_SECONDS),
    )


def renew(sessions, identifier, token):
    with sessions.begin() as session:
        result = session.execute(
            update(TrainingJob)
            .where(*owned(identifier, token))
            .values(heartbeat_at=func.clock_timestamp())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1


class Heartbeat:
    """Keep the lease alive during downloads, ML execution and artifact uploads."""

    def __init__(self, sessions, identifier, token):
        self.sessions, self.identifier, self.token = sessions, identifier, token
        self.stopped, self.lost = Event(), Event()
        self.last_success = time.monotonic()
        self.thread = Thread(target=self.run, name="training-heartbeat", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def run(self):
        while not self.stopped.wait(HEARTBEAT_SECONDS):
            try:
                if not renew(self.sessions, self.identifier, self.token):
                    self.lost.set()
                    return
                self.last_success = time.monotonic()
            except Exception:
                # Fail closed; raw connection errors must not reach public job errors.
                self.lost.set()
                return

    def check(self):
        # Also catch a heartbeat thread stuck in network I/O before the lease expires.
        if (
            self.lost.is_set()
            or time.monotonic() - self.last_success > LEASE_SECONDS / 2
        ):
            raise LeaseLost()

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=1)


def recover_stalled(sessions, *, limit=100):
    expired = (
        TrainingJob.status == "running",
        func.coalesce(
            TrainingJob.heartbeat_at, TrainingJob.started_at, TrainingJob.created_at
        )
        < func.clock_timestamp() - timedelta(seconds=LEASE_SECONDS),
    )
    with sessions() as session:
        identifiers = list(
            session.scalars(
                select(TrainingJob.id)
                .where(*expired)
                .order_by(TrainingJob.heartbeat_at.asc().nullsfirst(), TrainingJob.id)
                .limit(limit)
            )
        )
    recovered = 0
    for identifier in identifiers:
        with sessions.begin() as session:
            job = session.scalar(
                select(TrainingJob)
                .where(TrainingJob.id == identifier, *expired)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                continue
            # A lost heartbeat cannot prove that remote computation stopped.
            job.status, job.error = "failed", EXPIRED_ERROR
            job.finished_at = session.scalar(select(func.clock_timestamp()))
        recovered += 1
    return recovered
