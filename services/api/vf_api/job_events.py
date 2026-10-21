"""Public progress messages; raw worker output never enters the event channel."""

from datetime import datetime, timezone
import json
from uuid import UUID

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from vf_api.settings import QueueSettings

STAGES = {
    "dataset": "Checking the frozen dataset",
    "training": "Running training, test evaluation and ONNX verification",
    "artifacts": "Uploading verified model artifacts",
}
TERMINAL = {"succeeded", "failed", "cancelled"}


def channel(job_id):
    return f"job:{UUID(str(job_id))}:events"


def event(kind, job_id, **payload):
    return {
        "type": kind,
        "job_id": str(job_id),
        "ts": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def epoch_metrics(rows):
    return {
        "lr"
        if row["split"] == "train" and row["name"] == "learning_rate"
        else f"{row['split']}_{row['name']}": row["value"]
        for row in rows
    }


class JobPublisher:
    """Best-effort hints after database commits; a Redis failure cannot fail training."""

    def __init__(self, settings: QueueSettings | None = None):
        self.client = Redis.from_url(
            (settings or QueueSettings()).broker_url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            retry=Retry(NoBackoff(), 0),
        )
        self.available = True

    def publish(self, kind, job_id, **payload):
        if not self.available:
            return
        try:
            self.client.publish(
                channel(job_id),
                json.dumps(event(kind, job_id, **payload), allow_nan=False),
            )
        except RedisError:
            # Stop retrying for this job; durable state remains available to SSE readers.
            self.available = False

    def log(self, job_id, stage):
        self.publish("log", job_id, line=STAGES[stage])

    def close(self):
        self.client.close()
