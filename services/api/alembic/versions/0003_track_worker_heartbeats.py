"""Track worker ownership and recover lost dispatches.

Revision ID: 0003
Revises: 0002
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "training_jobs", sa.Column("heartbeat_at", sa.DateTime(timezone=True))
    )
    op.add_column("training_jobs", sa.Column("claim_token", sa.Uuid()))
    # Give pre-upgrade running rows one full recovery window.
    op.execute(
        "UPDATE training_jobs SET heartbeat_at = clock_timestamp() WHERE status = 'running'"
    )
    op.create_index(
        "ix_training_jobs_heartbeat",
        "training_jobs",
        ["heartbeat_at", "id"],
        postgresql_where=sa.text("status = 'running'"),
    )
    op.drop_index("ix_training_jobs_pending_dispatch", table_name="training_jobs")
    op.create_index(
        "ix_training_jobs_pending_dispatch",
        "training_jobs",
        ["dispatched_at", "created_at", "id"],
        postgresql_where=sa.text("status = 'queued'"),
    )


def downgrade():
    op.drop_index("ix_training_jobs_pending_dispatch", table_name="training_jobs")
    op.create_index(
        "ix_training_jobs_pending_dispatch",
        "training_jobs",
        ["created_at", "id"],
        postgresql_where=sa.text("status = 'queued' AND dispatched_at IS NULL"),
    )
    op.drop_index("ix_training_jobs_heartbeat", table_name="training_jobs")
    op.drop_column("training_jobs", "claim_token")
    op.drop_column("training_jobs", "heartbeat_at")
