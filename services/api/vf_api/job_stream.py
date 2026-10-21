"""Durable epoch catch-up with Redis wakeups and short database sessions."""

import asyncio
import json
import time

import anyio
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from vf_api.dependencies import token_user
from vf_api.job_events import STAGES, TERMINAL, channel, epoch_metrics, event
from vf_api.models import Model, Project, TrainingJob, TrainingMetric
from vf_api.settings import QueueSettings

POLL_SECONDS = 2.0
KEEPALIVE_SECONDS = 15.0
STREAM_SECONDS = 300.0
EPOCH_BATCH = 20
MAX_METRICS = 10000


def parse_cursor(value, job_id):
    if not value:
        return 0
    pieces = value.split(":")
    if (
        len(pieces) != 3
        or pieces[:2] != ["epoch", str(job_id)]
        or not pieces[2].isascii()
        or not pieces[2].isdecimal()
        or not 0 <= int(pieces[2]) <= 1000
    ):
        raise HTTPException(status_code=422, detail="Invalid job event cursor")
    return int(pieces[2])


def snapshot(sessions, settings, token, job_id, after, *, initial=False):
    with sessions() as session:
        session.execute(text("SET LOCAL statement_timeout = 5000"))
        user = token_user(token, "access", session, settings)
        job = session.scalar(
            select(TrainingJob)
            .join(Project, Project.id == TrainingJob.project_id)
            .where(TrainingJob.id == job_id, Project.owner_id == user.id)
        )
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        if initial and after:
            last = (
                session.scalar(
                    select(func.max(TrainingMetric.epoch)).where(
                        TrainingMetric.job_id == job_id
                    )
                )
                or 0
            )
            if after > last:
                raise HTTPException(
                    status_code=422,
                    detail="Job event cursor is ahead of recorded metrics",
                )
        model_id = (
            session.scalar(select(Model.id).where(Model.job_id == job_id))
            if job.status == "succeeded"
            else None
        )
        state = {
            "status": job.status,
            "gpu_id": job.gpu_id,
            "cancel_requested_at": job.cancel_requested_at.isoformat()
            if job.cancel_requested_at
            else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "error": job.error,
            "model_id": str(model_id) if model_id else None,
        }
        epochs = list(
            session.scalars(
                select(TrainingMetric.epoch)
                .where(TrainingMetric.job_id == job_id, TrainingMetric.epoch > after)
                .distinct()
                .order_by(TrainingMetric.epoch)
                .limit(EPOCH_BATCH + 1)
            )
        )
        rows = (
            session.scalars(
                select(TrainingMetric)
                .where(
                    TrainingMetric.job_id == job_id,
                    TrainingMetric.epoch.in_(epochs[:EPOCH_BATCH]),
                )
                .order_by(
                    TrainingMetric.epoch, TrainingMetric.split, TrainingMetric.name
                )
                .limit(MAX_METRICS + 1)
            ).all()
            if epochs
            else []
        )
        if len(rows) > MAX_METRICS:
            raise HTTPException(
                status_code=503, detail="Job metric batch exceeds stream limit"
            )
        grouped = {}
        for row in rows:
            grouped.setdefault(row.epoch, []).append(
                {"split": row.split, "name": row.name, "value": row.value}
            )
        return (
            state,
            [(epoch, epoch_metrics(values)) for epoch, values in grouped.items()],
            len(epochs) > EPOCH_BATCH,
        )


def encode(payload, *, cursor=None):
    prefix = f"id: {cursor}\n" if cursor else ""
    return f"{prefix}event: {payload['type']}\ndata: {json.dumps(payload, allow_nan=False)}\n\n"


def public_log(message, job_id):
    if not message or message.get("type") != "message":
        return None
    data = message.get("data")
    if not isinstance(data, (bytes, str)) or len(data) > 16384:
        return None
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("type") == "log"
        and value.get("job_id") == str(job_id)
        and value.get("line") in STAGES.values()
    ):
        return event("log", job_id, line=value["line"])
    return None


async def stream(sessions, settings, token, job_id, after, initial):
    client, subscription = None, None
    try:
        client = Redis.from_url(
            QueueSettings().broker_url,
            socket_connect_timeout=0.5,
            socket_timeout=3,
            retry=Retry(NoBackoff(), 0),
        )
        subscription = client.pubsub()
        try:
            await subscription.subscribe(channel(job_id))
        except RedisError:
            await subscription.aclose()
            subscription = None
        deadline = time.monotonic() + STREAM_SECONDS
        keepalive = time.monotonic() + KEEPALIVE_SECONDS
        previous, pending_log, current = None, None, initial
        yield "retry: 2000\n\n"
        while True:
            if current is None:
                try:
                    current = await run_in_threadpool(
                        snapshot, sessions, settings, token, job_id, after
                    )
                except HTTPException as error:
                    reason = {401: "authentication_required", 404: "job_not_found"}.get(
                        error.status_code, "stream_unavailable"
                    )
                    yield encode(event("error", job_id, reason=reason))
                    return
                except SQLAlchemyError:
                    yield encode(event("error", job_id, reason="database_unavailable"))
                    return
            state, epochs, more = current
            current = None
            if state != previous:
                yield encode(event("status", job_id, **state))
                previous = state
            if pending_log and state["status"] == "running":
                yield encode(pending_log)
            pending_log = None
            for epoch, metrics in epochs:
                yield encode(
                    event("epoch", job_id, epoch=epoch, metrics=metrics),
                    cursor=f"epoch:{job_id}:{epoch}",
                )
                after = epoch
            if not more and state["status"] in TERMINAL:
                yield encode(event("done", job_id, **state))
                return
            if time.monotonic() >= deadline:
                yield encode(event("reconnect", job_id, reason="stream_window_ended"))
                return
            if more:
                continue
            if time.monotonic() >= keepalive:
                yield ": keep-alive\n\n"
                keepalive = time.monotonic() + KEEPALIVE_SECONDS
            if subscription is None:
                await asyncio.sleep(POLL_SECONDS)
            else:
                try:
                    message = await subscription.get_message(
                        ignore_subscribe_messages=True, timeout=POLL_SECONDS
                    )
                    pending_log = public_log(message, job_id)
                    # A burst of notifications should not create an unbounded query loop.
                    if message:
                        await asyncio.sleep(0.05)
                except RedisError:
                    await subscription.aclose()
                    subscription = None
    finally:
        with anyio.move_on_after(2, shield=True):
            try:
                if subscription is not None:
                    await subscription.aclose()
            finally:
                if client is not None:
                    await client.aclose()


class JobStreamResponse(StreamingResponse):
    """Release the stream slot even if the peer disconnects before the first frame."""

    media_type = "text/event-stream"

    def __init__(self, content, slots):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )
        self.slots = slots

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                with anyio.move_on_after(2, shield=True):
                    await self.body_iterator.aclose()
            finally:
                self.slots.release()
