"""Retryable cleanup of projects whose deletion has been durably requested."""

from contextlib import ExitStack
import fcntl
import logging
from pathlib import Path
import shutil

from sqlalchemy import delete, select

from vf_api.models import (
    Annotation,
    Class,
    DatasetVersion,
    Deployment,
    Image,
    Model,
    Prediction,
    Project,
    TrainingJob,
    TrainingMetric,
)
from vf_api.routers.deployments import operation_lock

log = logging.getLogger(__name__)


def clean_project(identifier, sessions, stores, manager):
    with sessions() as session, ExitStack() as stack:
        project = session.scalar(
            select(Project)
            .where(Project.id == identifier, Project.deleting.is_(True))
            .with_for_update(skip_locked=True)
        )
        if project is None:
            return False
        if session.scalar(
            select(TrainingJob.id)
            .where(
                TrainingJob.project_id == identifier,
                TrainingJob.status.in_(["queued", "running"]),
            )
            .limit(1)
        ):
            return False  # The worker or lease recovery must acknowledge cancellation first.
        deployments = list(
            session.scalars(
                select(Deployment)
                .join(Model)
                .where(Model.project_id == identifier)
                .order_by(Deployment.id)
            )
        )
        for row in deployments:
            if not operation_lock(session, row.id):
                return False
        root = Path(manager.settings.prediction_log_dir).absolute()
        if root.exists():
            if root.is_symlink() or root.stat().st_mode & 0o077:
                raise ValueError("Invalid prediction directory")
            lock = stack.enter_context((root / ".collector.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        for row in deployments:
            manager.cleanup(
                {"id": row.id, "model_id": row.model_id, "project_id": identifier}
            )
            directory = root / str(row.id)
            if directory.is_symlink():
                raise ValueError("Invalid prediction directory")
            if directory.exists():
                shutil.rmtree(directory)
            stores["predictions"].delete_prefix(f"{row.id}/")
        for name in ("images", "datasets", "artifacts"):
            stores[name].delete_prefix(f"{identifier}/")
        stores["images"].delete_prefix(f"uploads/{identifier}/")
        stores["images"].delete_prefix(f"uploads/imports/{identifier}/")
        image_ids = select(Image.id).where(Image.project_id == identifier)
        job_ids = select(TrainingJob.id).where(TrainingJob.project_id == identifier)
        deployment_ids = [row.id for row in deployments]
        session.execute(
            delete(Prediction).where(Prediction.deployment_id.in_(deployment_ids))
        )
        session.execute(delete(Deployment).where(Deployment.id.in_(deployment_ids)))
        session.execute(delete(Model).where(Model.project_id == identifier))
        session.execute(
            delete(TrainingMetric).where(TrainingMetric.job_id.in_(job_ids))
        )
        session.execute(delete(TrainingJob).where(TrainingJob.project_id == identifier))
        session.execute(
            delete(DatasetVersion).where(DatasetVersion.project_id == identifier)
        )
        session.execute(delete(Annotation).where(Annotation.image_id.in_(image_ids)))
        session.execute(delete(Image).where(Image.project_id == identifier))
        session.execute(delete(Class).where(Class.project_id == identifier))
        session.delete(project)
        session.commit()
        return True


def clean_projects(sessions, stores, manager):
    with sessions() as session:
        identifiers = list(
            session.scalars(
                select(Project.id)
                .where(Project.deleting.is_(True))
                .order_by(Project.id)
                .limit(20)
            )
        )
    result = {"deleted": 0, "deferred": 0, "failed": 0}
    for identifier in identifiers:
        try:
            cleaned = clean_project(identifier, sessions, stores, manager)
        except Exception:
            # Preserve the request and continue without exposing private diagnostics.
            result["failed"] += 1
            log.warning("Project cleanup failed; request retained for retry: %s", identifier)
        else:
            result["deleted" if cleaned else "deferred"] += 1
    return result
