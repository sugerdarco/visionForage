"""Drain complete outbox events, preserving them until storage and DB commit."""

from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from vf_api.models import Deployment, Model, Prediction, Project
from vf_api.prediction_schemas import ClassificationOutput, PredictionEvent


def canonical(value):
    return str(UUID(value)) == value


def ingest_event(path, sessions, store):
    directory = path.parent
    if directory.is_symlink() or path.is_symlink() or path.stat().st_size > 256 * 1024:
        raise ValueError("Invalid prediction event")
    event = PredictionEvent.model_validate_json(path.read_bytes())
    if (
        str(event.prediction_id) != path.stem
        or str(event.deployment_id) != directory.name
    ):
        raise ValueError("Prediction identity mismatch")
    image = directory / f"{event.prediction_id}.image"
    with sessions() as session:
        # A crash after commit needs only local cleanup, even if image cleanup began.
        existing = session.get(Prediction, event.prediction_id)
        if existing is not None:
            if existing.deployment_id != event.deployment_id:
                raise ValueError("Prediction identity conflict")
        else:
            record = session.execute(
                select(Deployment, Model, Project.task_type)
                .join(Model, Model.id == Deployment.model_id)
                .join(Project, Project.id == Model.project_id)
                .where(Deployment.id == event.deployment_id)
            ).one_or_none()
            if record is None:
                raise ValueError("Unknown prediction deployment")
            deployment, model, task = record
            if (
                session.scalar(
                    select(Project.id)
                    .where(Project.id == model.project_id, Project.deleting.is_(False))
                    .with_for_update(read=True, key_share=True)
                )
                is None
            ):
                raise ValueError("Project deletion is in progress")
            if event.model_id != model.id or event.model_version != model.version:
                raise ValueError("Prediction model mismatch")
            if event.created_at > datetime.now(timezone.utc) + timedelta(minutes=5):
                raise ValueError("Prediction timestamp is in the future")
            output = event.output
            if isinstance(output, ClassificationOutput):
                if (
                    task != "classification"
                    or set(output.scores) != set(model.class_map)
                    or output.label not in output.scores
                    or output.confidence != output.scores[output.label]
                    or output.confidence != max(output.scores.values())
                ):
                    raise ValueError("Prediction classes mismatch")
                confidence = output.confidence
            else:
                if task != "detection" or any(
                    box.label not in model.class_map for box in output.boxes
                ):
                    raise ValueError("Prediction classes mismatch")
                confidence = max((box.confidence for box in output.boxes), default=0.0)
            if image.is_symlink() or image.stat().st_size != event.image_bytes:
                raise ValueError("Invalid prediction image")
            data = image.read_bytes()
            if hashlib.sha256(data).hexdigest() != event.image_sha256:
                raise ValueError("Prediction image checksum mismatch")
            suffix = "png" if event.mime == "image/png" else "jpg"
            date = event.created_at.astimezone(timezone.utc).date().isoformat()
            key = f"{deployment.id}/{date}/{event.prediction_id}.{suffix}"
            store.put_once(key, data, event.mime, event.image_sha256)
            session.add(
                Prediction(
                    id=event.prediction_id,
                    deployment_id=deployment.id,
                    image_key=key,
                    output=output.model_dump(),
                    max_confidence=confidence,
                    latency_ms=event.latency_ms,
                    sent_to_review=confidence < deployment.confidence_threshold,
                    created_at=event.created_at,
                )
            )
            session.commit()
    # Delete the event last so a crash during cleanup is replayable.
    image.unlink(missing_ok=True)
    path.unlink()


def collect_predictions(root, sessions, store, limit=100):
    root = Path(root).absolute()
    if not root.exists():
        return {"collected": 0, "invalid": 0}
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError("Prediction outbox parent must be private")
    result = {"collected": 0, "invalid": 0}
    with (root / ".collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return result
        for directory in sorted(root.iterdir()):
            try:
                valid = canonical(directory.name)
            except ValueError:
                continue
            if not valid or directory.is_symlink() or not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                if result["collected"] >= limit:
                    return result
                try:
                    ingest_event(path, sessions, store)
                except (ValueError, OSError):
                    # Retain invalid events for operator inspection, without logging paths/data.
                    result["invalid"] += 1
                    continue
                result["collected"] += 1
    return result
