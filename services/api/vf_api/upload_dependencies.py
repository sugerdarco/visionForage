"""Storage and queue clients are initialized only when upload routes need them."""

from typing import Annotated

from celery import Celery
from fastapi import Depends, Request

from vf_api.queueing import make_queue
from vf_api.settings import StorageSettings
from vf_api.storage import ArtifactStore, DatasetStore, ObjectStore


def get_store(request: Request) -> ObjectStore:
    with request.app.state.resource_lock:
        if not hasattr(request.app.state, "store"):
            request.app.state.store = ObjectStore(StorageSettings())
    return request.app.state.store


def get_queue(request: Request) -> Celery:
    with request.app.state.resource_lock:
        if not hasattr(request.app.state, "queue"):
            request.app.state.queue = make_queue()
    return request.app.state.queue


def get_datasets(request: Request) -> DatasetStore:
    with request.app.state.resource_lock:
        if not hasattr(request.app.state, "datasets"):
            request.app.state.datasets = DatasetStore(StorageSettings())
    return request.app.state.datasets


def get_artifacts(request: Request) -> ArtifactStore:
    with request.app.state.resource_lock:
        if not hasattr(request.app.state, "artifacts"):
            request.app.state.artifacts = ArtifactStore(StorageSettings())
    return request.app.state.artifacts


Storage = Annotated[ObjectStore, Depends(get_store)]
Queue = Annotated[Celery, Depends(get_queue)]
Datasets = Annotated[DatasetStore, Depends(get_datasets)]
Artifacts = Annotated[ArtifactStore, Depends(get_artifacts)]
