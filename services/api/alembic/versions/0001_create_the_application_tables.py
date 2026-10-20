"""Create the application tables"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "projects",
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("task_type", sa.String(length=20), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "task_type IN ('classification', 'detection', 'anomaly')",
            name=op.f("ck_projects_task_type"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_projects_owner_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
    )
    op.create_index(
        op.f("ix_projects_owner_id"), "projects", ["owner_id"], unique=False
    )
    op.create_table(
        "classes",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "color ~ '^#[0-9a-fA-F]{6}$'", name=op.f("ck_classes_color")
        ),
        sa.CheckConstraint('"index" >= 0', name=op.f("ck_classes_nonnegative_index")),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_classes_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_classes")),
        sa.UniqueConstraint("project_id", "index", name="uq_classes_project_index"),
        sa.UniqueConstraint("project_id", "name", name="uq_classes_project_name"),
    )
    op.create_table(
        "dataset_versions",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("manifest_key", sa.Text(), nullable=False),
        sa.Column(
            "split_ratios", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("counts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(split_ratios) = 'object' AND jsonb_typeof(counts) = 'object'",
            name=op.f("ck_dataset_versions_json_objects"),
        ),
        sa.CheckConstraint(
            "version > 0", name=op.f("ck_dataset_versions_positive_version")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_dataset_versions_project_id_projects"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_versions")),
        sa.UniqueConstraint("id", "project_id", name="uq_dataset_versions_id_project"),
        sa.UniqueConstraint(
            "manifest_key", name=op.f("uq_dataset_versions_manifest_key")
        ),
        sa.UniqueConstraint(
            "project_id", "version", name="uq_dataset_versions_project_version"
        ),
    )
    op.create_table(
        "images",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("thumb_key", sa.Text(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=20), server_default="unlabeled", nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_images_sha256")),
        sa.CheckConstraint(
            "status IN ('unlabeled', 'labeled', 'in_review')",
            name=op.f("ck_images_status"),
        ),
        sa.CheckConstraint(
            "width > 0 AND height > 0", name=op.f("ck_images_dimensions")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_images_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_images")),
        sa.UniqueConstraint("project_id", "sha256", name="uq_images_project_sha256"),
        sa.UniqueConstraint("storage_key", name=op.f("uq_images_storage_key")),
    )
    op.create_index(
        "ix_images_project_status", "images", ["project_id", "status"], unique=False
    )
    op.create_table(
        "annotations",
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("class_id", sa.Uuid(), nullable=False),
        sa.Column("bbox", postgresql.ARRAY(sa.Double()), nullable=True),
        sa.Column(
            "source", sa.String(length=10), server_default="human", nullable=False
        ),
        sa.Column("confidence", sa.Double(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source IN ('human', 'model')", name=op.f("ck_annotations_source")
        ),
        sa.CheckConstraint(
            "bbox IS NULL OR (array_ndims(bbox) = 1 AND array_lower(bbox, 1) = 1 AND cardinality(bbox) = 4 AND bbox[1] IS NOT NULL AND bbox[2] IS NOT NULL AND bbox[3] IS NOT NULL AND bbox[4] IS NOT NULL AND bbox[1] BETWEEN 0 AND 1 AND bbox[2] BETWEEN 0 AND 1 AND bbox[3] > 0 AND bbox[3] <= 1 AND bbox[4] > 0 AND bbox[4] <= 1 AND bbox[1] + bbox[3] <= 1.000000001 AND bbox[2] + bbox[4] <= 1.000000001)",
            name=op.f("ck_annotations_normalized_bbox"),
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0 AND 1", name=op.f("ck_annotations_confidence")
        ),
        sa.ForeignKeyConstraint(
            ["class_id"], ["classes.id"], name=op.f("fk_annotations_class_id_classes")
        ),
        sa.ForeignKeyConstraint(
            ["image_id"], ["images.id"], name=op.f("fk_annotations_image_id_images")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_annotations")),
    )
    op.create_index(
        op.f("ix_annotations_class_id"), "annotations", ["class_id"], unique=False
    )
    op.create_index(
        op.f("ix_annotations_image_id"), "annotations", ["image_id"], unique=False
    )
    op.create_table(
        "training_jobs",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version_id", sa.Uuid(), nullable=False),
        sa.Column("model_arch", sa.String(length=100), nullable=False),
        sa.Column(
            "hyperparams",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status", sa.String(length=20), server_default="queued", nullable=False
        ),
        sa.Column("gpu_id", sa.Integer(), nullable=True),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(hyperparams) = 'object'",
            name=op.f("ck_training_jobs_hyperparams_object"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_training_jobs_status"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name=op.f("ck_training_jobs_time_order"),
        ),
        sa.CheckConstraint("gpu_id >= 0", name=op.f("ck_training_jobs_gpu_id")),
        sa.ForeignKeyConstraint(
            ["dataset_version_id", "project_id"],
            ["dataset_versions.id", "dataset_versions.project_id"],
            name=op.f("fk_training_jobs_dataset_version_id_dataset_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_training_jobs_project_id_projects"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_jobs")),
        sa.UniqueConstraint(
            "celery_task_id", name=op.f("uq_training_jobs_celery_task_id")
        ),
        sa.UniqueConstraint("id", "project_id", name="uq_training_jobs_id_project"),
    )
    op.create_index(
        op.f("ix_training_jobs_dataset_version_id"),
        "training_jobs",
        ["dataset_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_jobs_project_id"),
        "training_jobs",
        ["project_id"],
        unique=False,
    )
    op.create_table(
        "models",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("weights_key", sa.Text(), nullable=False),
        sa.Column("onnx_key", sa.Text(), nullable=False),
        sa.Column("input_size", sa.Integer(), nullable=False),
        sa.Column("class_map", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "is_promoted", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(metrics) = 'object' AND jsonb_typeof(class_map) = 'object'",
            name=op.f("ck_models_json_objects"),
        ),
        sa.CheckConstraint(
            "version > 0 AND input_size > 0", name=op.f("ck_models_positive_dimensions")
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "project_id"],
            ["training_jobs.id", "training_jobs.project_id"],
            name=op.f("fk_models_job_id_training_jobs"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name=op.f("fk_models_project_id_projects")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_models")),
        sa.UniqueConstraint("job_id", name=op.f("uq_models_job_id")),
        sa.UniqueConstraint("project_id", "version", name="uq_models_project_version"),
    )
    op.create_index(
        "uq_models_one_promoted",
        "models",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("is_promoted"),
    )
    op.create_table(
        "training_metrics",
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("split", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("value", sa.Double(), nullable=False),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "split IN ('train', 'val')", name=op.f("ck_training_metrics_split")
        ),
        sa.CheckConstraint(
            "value > '-Infinity'::float8 AND value < 'Infinity'::float8",
            name=op.f("ck_training_metrics_finite_value"),
        ),
        sa.CheckConstraint("epoch >= 0", name=op.f("ck_training_metrics_epoch")),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["training_jobs.id"],
            name=op.f("fk_training_metrics_job_id_training_jobs"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_metrics")),
        sa.UniqueConstraint(
            "job_id", "epoch", "split", "name", name="uq_training_metrics_epoch_value"
        ),
    )
    op.create_index(
        "ix_training_metrics_job_name_epoch",
        "training_metrics",
        ["job_id", "name", "epoch"],
        unique=False,
    )
    op.create_table(
        "deployments",
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("container_id", sa.String(length=255), nullable=True),
        sa.Column(
            "status", sa.String(length=20), server_default="starting", nullable=False
        ),
        sa.Column("api_key_hash", sa.Text(), nullable=False),
        sa.Column(
            "confidence_threshold", sa.Double(), server_default="0.6", nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('starting', 'running', 'stopped', 'failed')",
            name=op.f("ck_deployments_status"),
        ),
        sa.CheckConstraint(
            "confidence_threshold BETWEEN 0 AND 1",
            name=op.f("ck_deployments_confidence_threshold"),
        ),
        sa.ForeignKeyConstraint(
            ["model_id"], ["models.id"], name=op.f("fk_deployments_model_id_models")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_deployments")),
        sa.UniqueConstraint("slug", name=op.f("uq_deployments_slug")),
    )
    op.create_index(
        op.f("ix_deployments_model_id"), "deployments", ["model_id"], unique=False
    )
    op.create_table(
        "predictions",
        sa.Column("deployment_id", sa.Uuid(), nullable=False),
        sa.Column("image_key", sa.Text(), nullable=False),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("max_confidence", sa.Double(), nullable=False),
        sa.Column("latency_ms", sa.Double(), nullable=False),
        sa.Column(
            "sent_to_review",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(output) = 'object'", name=op.f("ck_predictions_output_object")
        ),
        sa.CheckConstraint(
            "latency_ms >= 0 AND latency_ms < 'Infinity'::float8",
            name=op.f("ck_predictions_latency_ms"),
        ),
        sa.CheckConstraint(
            "max_confidence BETWEEN 0 AND 1", name=op.f("ck_predictions_max_confidence")
        ),
        sa.ForeignKeyConstraint(
            ["deployment_id"],
            ["deployments.id"],
            name=op.f("fk_predictions_deployment_id_deployments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )
    op.create_index(
        "ix_predictions_deployment_created",
        "predictions",
        ["deployment_id", "created_at"],
        unique=False,
    )

    op.execute("""
        CREATE FUNCTION vf_touch_updated_at() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = statement_timestamp();
            RETURN NEW;
        END;
        $$
    """)
    for table in (
        "users",
        "projects",
        "classes",
        "images",
        "annotations",
        "training_jobs",
        "models",
        "deployments",
        "predictions",
    ):
        op.execute(
            f"CREATE TRIGGER touch_updated_at BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION vf_touch_updated_at()"
        )
    op.execute("""
        CREATE FUNCTION vf_reject_dataset_version_update() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'Dataset versions cannot be updated; create a new version'
                USING ERRCODE = '23514';
        END;
        $$
    """)
    op.execute(
        "CREATE TRIGGER immutable_dataset_version BEFORE UPDATE ON dataset_versions "
        "FOR EACH ROW EXECUTE FUNCTION vf_reject_dataset_version_update()"
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_deployment_created", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_deployments_model_id"), table_name="deployments")
    op.drop_table("deployments")
    op.drop_index("ix_training_metrics_job_name_epoch", table_name="training_metrics")
    op.drop_table("training_metrics")
    op.drop_index(
        "uq_models_one_promoted",
        table_name="models",
        postgresql_where=sa.text("is_promoted"),
    )
    op.drop_table("models")
    op.drop_index(op.f("ix_training_jobs_project_id"), table_name="training_jobs")
    op.drop_index(
        op.f("ix_training_jobs_dataset_version_id"), table_name="training_jobs"
    )
    op.drop_table("training_jobs")
    op.drop_index(op.f("ix_annotations_image_id"), table_name="annotations")
    op.drop_index(op.f("ix_annotations_class_id"), table_name="annotations")
    op.drop_table("annotations")
    op.drop_index("ix_images_project_status", table_name="images")
    op.drop_table("images")
    op.drop_table("dataset_versions")
    op.drop_table("classes")
    op.drop_index(op.f("ix_projects_owner_id"), table_name="projects")
    op.drop_table("projects")
    op.drop_table("users")
    op.execute("DROP FUNCTION vf_reject_dataset_version_update()")
    op.execute("DROP FUNCTION vf_touch_updated_at()")
