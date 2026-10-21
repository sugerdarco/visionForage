"""A solo Celery supervisor; the selected Python subprocess owns CUDA."""

from vf_api.database import make_engine, make_session_factory
from vf_api.queueing import TRAIN_TASK, make_queue
from vf_api.settings import Settings, StorageSettings
from vf_api.storage import ArtifactStore, DatasetStore, ObjectStore
from vf_worker.settings import WorkerSettings
from vf_worker.training import WorkerBusy, run_job


def create_gpu_worker(database=None, storage=None, queue=None, worker=None):
    app = make_queue(queue)

    @app.task(name=TRAIN_TASK, bind=True, shared=False, lazy=False, max_retries=None)
    def train(self, job_id):
        configuration = worker or WorkerSettings()
        engine = make_engine(database or Settings())
        storage_settings = storage or StorageSettings()
        stores = []
        try:
            for store_class in (DatasetStore, ObjectStore, ArtifactStore):
                stores.append(store_class(storage_settings))
            return run_job(
                job_id,
                self.request.id,
                make_session_factory(engine),
                configuration,
                *stores,
            )
        except WorkerBusy:
            raise self.retry(countdown=10) from None
        finally:
            for store in stores:
                store.close()
            engine.dispose()

    return app


if __name__ == "__main__":
    configuration = WorkerSettings()
    if not configuration.worker_ml_python.is_file():
        raise SystemExit(
            "WORKER_ML_PYTHON must point to the existing ML Python executable"
        )
    app = create_gpu_worker(worker=configuration)
    app.worker_main(
        [
            "worker",
            "--pool=solo",
            "--concurrency=1",
            "--queues=gpu",
            "--loglevel=INFO",
            f"--hostname=gpu-{configuration.worker_gpu_id}@%h",
            "--without-gossip",
            "--without-mingle",
        ]
    )
