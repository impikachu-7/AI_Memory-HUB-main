"""add persisted per-user settings

Revision ID: 20260922_08
Revises: 20260914_07
"""
from alembic import op
import sqlalchemy as sa
import uuid

revision = "20260922_08"
down_revision = "20260914_07"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_settings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("memory_retrieval_mode", sa.String(32), nullable=False, server_default="automatic"),
        sa.Column("notifications_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", name="uq_user_settings_user_id"),
    )

    conn = op.get_bind()
    users = conn.execute(sa.text("SELECT id FROM users"))
    conn.execute(
        sa.text("INSERT INTO user_settings (id, user_id, memory_enabled, memory_retrieval_mode, notifications_enabled) VALUES (:id, :user_id, true, 'automatic', true)"),
        [{"id": str(uuid.uuid4()), "user_id": row[0]} for row in users],
    )


def downgrade():
    op.drop_table("user_settings")