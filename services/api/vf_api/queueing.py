"""JSON-only CPU queue; thumbnail state remains durable in PostgreSQL."""

from celery import Celery

from vf_api.settings import QueueSettings

THUMBNAIL_TASK = "visionforge.thumbnails.create"
RECOVERY_TASK = "visionforge.thumbnails.recover"


def make_queue(settings: QueueSettings | None = None) -> Celery:
    settings = settings or QueueSettings()
    app = Celery("visionforge", broker=settings.broker_url)
    app.conf.update(
        task_default_queue="cpu",
        task_serializer="json",
        accept_content=["json"],
        task_ignore_result=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=3,
        broker_transport_options={
            "global_keyprefix": "visionforge:",
            "socket_connect_timeout": 3,
            "socket_timeout": 3,
            "visibility_timeout": 300,
        },
        task_publish_retry=False,
        beat_schedule={
            "recover-missing-thumbnails": {
                "task": RECOVERY_TASK,
                "schedule": 60.0,
            },
        },
    )
    return app
