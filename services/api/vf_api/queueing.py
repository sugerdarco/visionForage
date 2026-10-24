"""JSON-only queues; durable work state remains in PostgreSQL."""

from celery import Celery

from vf_api.settings import QueueSettings

THUMBNAIL_TASK = "visionforge.thumbnails.create"
RECOVERY_TASK = "visionforge.thumbnails.recover"
TRAIN_TASK = "visionforge.training.run"
DISPATCH_TASK = "visionforge.training.dispatch"
PREDICTION_TASK = "visionforge.predictions.collect"
PROJECT_CLEANUP_TASK = "visionforge.projects.cleanup"


def make_queue(settings: QueueSettings | None = None) -> Celery:
    settings = settings or QueueSettings()
    app = Celery("visionforge", broker=settings.broker_url)
    app.conf.update(
        task_default_queue="cpu",
        task_routes={TRAIN_TASK: {"queue": "gpu"}},
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
            "cleanup-deleted-projects": {
                "task": PROJECT_CLEANUP_TASK,
                "schedule": 10.0,
            },
            "collect-predictions": {"task": PREDICTION_TASK, "schedule": 5.0},
            "dispatch-training-jobs": {
                "task": DISPATCH_TASK,
                "schedule": 10.0,
            },
            "recover-missing-thumbnails": {
                "task": RECOVERY_TASK,
                "schedule": 60.0,
            },
        },
    )
    return app
