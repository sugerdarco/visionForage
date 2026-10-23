"""Idempotent thumbnails and recovery of committed images without thumbnails."""

import hashlib
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError

from vf_api.database import make_engine, make_session_factory
from vf_api.image_files import make_thumbnail
from vf_api.models import Image
from vf_api.queueing import DISPATCH_TASK, RECOVERY_TASK, THUMBNAIL_TASK, make_queue
from vf_api.settings import QueueSettings, Settings, StorageSettings
from vf_api.storage import ObjectStore
from vf_worker.dispatch import dispatch_pending
from vf_worker.leases import recover_stalled
from vf_worker.predictions import collect_predictions
from vf_api.queueing import PREDICTION_TASK
from vf_api.settings import DeploymentSettings
from vf_api.storage import PredictionStore


def create_worker(
    database: Settings | None = None,
    storage: StorageSettings | None = None,
    queue: QueueSettings | None = None,
    deployments: DeploymentSettings | None = None,
):
    database = database or Settings()
    storage = storage or StorageSettings()
    app = make_queue(queue)

    @app.task(name=PREDICTION_TASK, shared=False, lazy=False)
    def predictions():
        engine = make_engine(database)
        store = PredictionStore(storage)
        try:
            return collect_predictions(
                (deployments or DeploymentSettings()).prediction_log_dir,
                make_session_factory(engine),
                store,
            )
        finally:
            store.close()
            engine.dispose()

    @app.task(name=DISPATCH_TASK, shared=False, lazy=False)
    def dispatch():
        engine = make_engine(database)
        try:
            sessions = make_session_factory(engine)
            recover_stalled(sessions)
            return dispatch_pending(sessions, app)
        finally:
            engine.dispose()

    @app.task(
        name=THUMBNAIL_TASK,
        shared=False,
        lazy=False,
        autoretry_for=(ClientError, BotoCoreError, OperationalError),
        retry_backoff=5,
        retry_backoff_max=60,
        max_retries=5,
    )
    def thumbnail(image_id: str):
        engine = make_engine(database)
        store = ObjectStore(storage)
        try:
            with make_session_factory(engine)() as session:
                image = session.get(Image, UUID(image_id))
                if image is None or image.thumb_key is not None:
                    return
                data = store.read(image.storage_key)
                if hashlib.sha256(data).hexdigest() != image.sha256:
                    raise ValueError("Original image checksum no longer matches")
                preview = make_thumbnail(data)
                key = f"{image.project_id}/thumbs/{image.sha256}-v1.jpg"
                store.put(key, preview, "image/jpeg")
                # Duplicate delivery writes the same object and makes the same update.
                session.execute(
                    update(Image)
                    .where(Image.id == image.id, Image.thumb_key.is_(None))
                    .values(thumb_key=key)
                )
                session.commit()
        finally:
            store.close()
            engine.dispose()

    @app.task(name=RECOVERY_TASK, shared=False, lazy=False)
    def recover():
        engine = make_engine(database)
        try:
            with make_session_factory(engine)() as session:
                pending = session.scalars(
                    select(Image.id)
                    .where(Image.thumb_key.is_(None))
                    .order_by(Image.created_at, Image.id)
                    .execution_options(yield_per=100)
                )
                for identifier in pending:
                    app.send_task(THUMBNAIL_TASK, args=[str(identifier)], retry=False)
        finally:
            engine.dispose()

    return app


app = create_worker()
