"""Track training dispatch and cooperative cancellation.

Revision ID: 0002
Revises: 0001
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "training_jobs", sa.Column("dispatched_at", sa.DateTime(timezone=True))
    )
    op.add_column(
        "training_jobs", sa.Column("cancel_requested_at", sa.DateTime(timezone=True))
    )
    op.create_index(
        "ix_training_jobs_pending_dispatch",
        "training_jobs",
        ["created_at", "id"],
        postgresql_where=sa.text("status = 'queued' AND dispatched_at IS NULL"),
    )


def downgrade():
    op.drop_index("ix_training_jobs_pending_dispatch", table_name="training_jobs")
    op.drop_column("training_jobs", "cancel_requested_at")
    op.drop_column("training_jobs", "dispatched_at")
