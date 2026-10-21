"""Publish committed jobs; ambiguous broker acknowledgements may produce duplicates."""

from datetime import datetime, timezone

from sqlalchemy import select

from vf_api.models import TrainingJob
from vf_api.queueing import TRAIN_TASK


def dispatch_pending(sessions, queue, *, limit=100):
    pending = (
        TrainingJob.status == "queued",
        TrainingJob.dispatched_at.is_(None),
        TrainingJob.cancel_requested_at.is_(None),
        TrainingJob.celery_task_id.is_not(None),
    )
    with sessions() as session:
        identifiers = list(
            session.scalars(
                select(TrainingJob.id)
                .where(*pending)
                .order_by(TrainingJob.created_at, TrainingJob.id)
                .limit(limit)
            )
        )
    published = 0
    for identifier in identifiers:
        with sessions.begin() as session:
            job = session.scalar(
                select(TrainingJob)
                .where(TrainingJob.id == identifier, *pending)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                continue
            # Lock covers send + commit, so cancellation and other dispatchers serialize.
            # Never mark sent before publishing: broker failure must leave a retryable row.
            queue.send_task(
                TRAIN_TASK,
                args=[str(job.id)],
                task_id=job.celery_task_id,
                retry=False,
            )
            job.dispatched_at = datetime.now(timezone.utc)
        published += 1
    return published
