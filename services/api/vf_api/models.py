"""The build plan's metadata tables. Binary artifacts live in object storage."""

from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Coordinate(sa.TypeDecorator[float]):
    """Give psycopg homogeneous float arrays even when a box includes integer zeros."""

    impl = sa.Double
    cache_ok = True

    def process_bind_param(
        self, value: float | int | None, dialect: sa.engine.Dialect
    ) -> float | None:
        return None if value is None else float(value)


class Base(DeclarativeBase):
    metadata = sa.MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Record:
    id: Mapped[UUID] = mapped_column(
        sa.Uuid, primary_key=True, server_default=sa.text("gen_random_uuid()")
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )


class MutableRecord(Record):
    # The migration installs a database trigger so raw SQL updates also touch this.
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )


class User(MutableRecord, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(sa.String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(sa.Text)


class Project(MutableRecord, Base):
    __tablename__ = "projects"
    __table_args__ = (
        sa.CheckConstraint(
            "task_type IN ('classification', 'detection', 'anomaly')", name="task_type"
        ),
    )
    owner_id: Mapped[UUID] = mapped_column(sa.ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(sa.String(200))
    task_type: Mapped[str] = mapped_column(sa.String(20))


class Class(MutableRecord, Base):
    __tablename__ = "classes"
    __table_args__ = (
        sa.UniqueConstraint("project_id", "name", name="uq_classes_project_name"),
        sa.UniqueConstraint("project_id", "index", name="uq_classes_project_index"),
        sa.CheckConstraint('"index" >= 0', name="nonnegative_index"),
        sa.CheckConstraint("color ~ '^#[0-9a-fA-F]{6}$'", name="color"),
    )
    project_id: Mapped[UUID] = mapped_column(sa.ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(sa.String(200))
    color: Mapped[str] = mapped_column(sa.String(7))
    index: Mapped[int] = mapped_column(sa.Integer)


class Image(MutableRecord, Base):
    __tablename__ = "images"
    __table_args__ = (
        sa.UniqueConstraint("project_id", "sha256", name="uq_images_project_sha256"),
        sa.CheckConstraint("width > 0 AND height > 0", name="dimensions"),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256"),
        sa.CheckConstraint(
            "status IN ('unlabeled', 'labeled', 'in_review')", name="status"
        ),
        sa.Index("ix_images_project_status", "project_id", "status"),
    )
    project_id: Mapped[UUID] = mapped_column(sa.ForeignKey("projects.id"))
    storage_key: Mapped[str] = mapped_column(sa.Text, unique=True)
    thumb_key: Mapped[str | None] = mapped_column(sa.Text)
    width: Mapped[int] = mapped_column(sa.Integer)
    height: Mapped[int] = mapped_column(sa.Integer)
    sha256: Mapped[str] = mapped_column(sa.String(64))
    status: Mapped[str] = mapped_column(sa.String(20), server_default="unlabeled")


class Annotation(MutableRecord, Base):
    __tablename__ = "annotations"
    __table_args__ = (
        sa.CheckConstraint("source IN ('human', 'model')", name="source"),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence"),
        sa.CheckConstraint(
            "bbox IS NULL OR (array_ndims(bbox) = 1 AND array_lower(bbox, 1) = 1 "
            "AND cardinality(bbox) = 4 "
            "AND bbox[1] IS NOT NULL AND bbox[2] IS NOT NULL "
            "AND bbox[3] IS NOT NULL AND bbox[4] IS NOT NULL "
            "AND bbox[1] BETWEEN 0 AND 1 AND bbox[2] BETWEEN 0 AND 1 "
            "AND bbox[3] > 0 AND bbox[3] <= 1 AND bbox[4] > 0 AND bbox[4] <= 1 "
            "AND bbox[1] + bbox[3] <= 1.000000001 AND bbox[2] + bbox[4] <= 1.000000001)",
            name="normalized_bbox",
        ),
    )
    image_id: Mapped[UUID] = mapped_column(sa.ForeignKey("images.id"), index=True)
    class_id: Mapped[UUID] = mapped_column(sa.ForeignKey("classes.id"), index=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Coordinate))
    source: Mapped[str] = mapped_column(sa.String(10), server_default="human")
    confidence: Mapped[float | None] = mapped_column(sa.Double)


class DatasetVersion(Record, Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        sa.UniqueConstraint(
            "project_id", "version", name="uq_dataset_versions_project_version"
        ),
        sa.UniqueConstraint("id", "project_id", name="uq_dataset_versions_id_project"),
        sa.CheckConstraint("version > 0", name="positive_version"),
        sa.CheckConstraint(
            "jsonb_typeof(split_ratios) = 'object' AND jsonb_typeof(counts) = 'object'",
            name="json_objects",
        ),
    )
    project_id: Mapped[UUID] = mapped_column(sa.ForeignKey("projects.id"))
    version: Mapped[int] = mapped_column(sa.Integer)
    manifest_key: Mapped[str] = mapped_column(sa.Text, unique=True)
    split_ratios: Mapped[dict[str, Any]] = mapped_column(JSONB)
    counts: Mapped[dict[str, Any]] = mapped_column(JSONB)


class TrainingJob(MutableRecord, Base):
    __tablename__ = "training_jobs"
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
        ),
        sa.UniqueConstraint("id", "project_id", name="uq_training_jobs_id_project"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="status",
        ),
        sa.CheckConstraint("gpu_id >= 0", name="gpu_id"),
        sa.CheckConstraint(
            "jsonb_typeof(hyperparams) = 'object'", name="hyperparams_object"
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name="time_order",
        ),
    )
    project_id: Mapped[UUID] = mapped_column(sa.ForeignKey("projects.id"), index=True)
    dataset_version_id: Mapped[UUID] = mapped_column(sa.Uuid, index=True)
    model_arch: Mapped[str] = mapped_column(sa.String(100))
    hyperparams: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=sa.text("'{}'::jsonb")
    )
    status: Mapped[str] = mapped_column(sa.String(20), server_default="queued")
    gpu_id: Mapped[int | None] = mapped_column(sa.Integer)
    celery_task_id: Mapped[str | None] = mapped_column(sa.String(255), unique=True)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(sa.Text)


class TrainingMetric(Record, Base):
    __tablename__ = "training_metrics"
    __table_args__ = (
        sa.UniqueConstraint(
            "job_id", "epoch", "split", "name", name="uq_training_metrics_epoch_value"
        ),
        sa.CheckConstraint("epoch >= 0", name="epoch"),
        sa.CheckConstraint("split IN ('train', 'val')", name="split"),
        sa.CheckConstraint(
            "value > '-Infinity'::float8 AND value < 'Infinity'::float8",
            name="finite_value",
        ),
        sa.Index("ix_training_metrics_job_name_epoch", "job_id", "name", "epoch"),
    )
    job_id: Mapped[UUID] = mapped_column(sa.ForeignKey("training_jobs.id"))
    epoch: Mapped[int] = mapped_column(sa.Integer)
    split: Mapped[str] = mapped_column(sa.String(10))
    name: Mapped[str] = mapped_column(sa.String(100))
    value: Mapped[float] = mapped_column(sa.Double)


class Model(MutableRecord, Base):
    __tablename__ = "models"
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["job_id", "project_id"], ["training_jobs.id", "training_jobs.project_id"]
        ),
        sa.UniqueConstraint("project_id", "version", name="uq_models_project_version"),
        sa.CheckConstraint(
            "version > 0 AND input_size > 0", name="positive_dimensions"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(metrics) = 'object' AND jsonb_typeof(class_map) = 'object'",
            name="json_objects",
        ),
        sa.Index(
            "uq_models_one_promoted",
            "project_id",
            unique=True,
            postgresql_where=sa.text("is_promoted"),
        ),
    )
    project_id: Mapped[UUID] = mapped_column(sa.ForeignKey("projects.id"))
    job_id: Mapped[UUID] = mapped_column(sa.Uuid, unique=True)
    version: Mapped[int] = mapped_column(sa.Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    weights_key: Mapped[str] = mapped_column(sa.Text)
    onnx_key: Mapped[str] = mapped_column(sa.Text)
    input_size: Mapped[int] = mapped_column(sa.Integer)
    class_map: Mapped[dict[str, Any]] = mapped_column(JSONB)
    is_promoted: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false())


class Deployment(MutableRecord, Base):
    __tablename__ = "deployments"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('starting', 'running', 'stopped', 'failed')", name="status"
        ),
        sa.CheckConstraint(
            "confidence_threshold BETWEEN 0 AND 1", name="confidence_threshold"
        ),
    )
    model_id: Mapped[UUID] = mapped_column(sa.ForeignKey("models.id"), index=True)
    slug: Mapped[str] = mapped_column(sa.String(100), unique=True)
    container_id: Mapped[str | None] = mapped_column(sa.String(255))
    status: Mapped[str] = mapped_column(sa.String(20), server_default="starting")
    api_key_hash: Mapped[str] = mapped_column(sa.Text)
    confidence_threshold: Mapped[float] = mapped_column(sa.Double, server_default="0.6")


class Prediction(MutableRecord, Base):
    __tablename__ = "predictions"
    __table_args__ = (
        sa.CheckConstraint("max_confidence BETWEEN 0 AND 1", name="max_confidence"),
        sa.CheckConstraint(
            "latency_ms >= 0 AND latency_ms < 'Infinity'::float8", name="latency_ms"
        ),
        sa.CheckConstraint("jsonb_typeof(output) = 'object'", name="output_object"),
        sa.Index("ix_predictions_deployment_created", "deployment_id", "created_at"),
    )
    deployment_id: Mapped[UUID] = mapped_column(sa.ForeignKey("deployments.id"))
    image_key: Mapped[str] = mapped_column(sa.Text)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    max_confidence: Mapped[float] = mapped_column(sa.Double)
    latency_ms: Mapped[float] = mapped_column(sa.Double)
    sent_to_review: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.false())
