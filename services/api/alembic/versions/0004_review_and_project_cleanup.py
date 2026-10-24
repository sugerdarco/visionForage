"""Track accepted reviews and durable project deletion requests."""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "projects",
        sa.Column("deleting", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("predictions", sa.Column("review_image_id", sa.Uuid(), nullable=True))
    op.add_column(
        "predictions",
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "predictions", sa.Column("review_payload_sha256", sa.String(64), nullable=True)
    )
    op.create_foreign_key(
        "fk_predictions_review_image_id_images",
        "predictions",
        "images",
        ["review_image_id"],
        ["id"],
    )
    op.create_check_constraint(
        op.f("ck_predictions_review_receipt"),
        "predictions",
        "(review_image_id IS NULL AND reviewed_at IS NULL AND review_payload_sha256 IS NULL) OR (review_image_id IS NOT NULL AND reviewed_at IS NOT NULL AND review_payload_sha256 IS NOT NULL AND review_payload_sha256 ~ '^[a-f0-9]{64}$')",
    )


def downgrade():
    op.drop_constraint(
        op.f("ck_predictions_review_receipt"), "predictions", type_="check"
    )
    op.drop_constraint(
        "fk_predictions_review_image_id_images", "predictions", type_="foreignkey"
    )
    op.drop_column("predictions", "review_payload_sha256")
    op.drop_column("predictions", "reviewed_at")
    op.drop_column("predictions", "review_image_id")
    op.drop_column("projects", "deleting")
