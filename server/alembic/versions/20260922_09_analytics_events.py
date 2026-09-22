"""add user-scoped analytics events

Revision ID: 20260922_09
Revises: 20260922_08
"""
from alembic import op
import sqlalchemy as sa

revision = "20260922_09"
down_revision = "20260922_08"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "analytics_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(50)),
        sa.Column("model_key", sa.String(150)),
        sa.Column("is_local", sa.Boolean()),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_analytics_events_user_id", "analytics_events", ["user_id"])
    op.create_index("ix_analytics_events_event_type", "analytics_events", ["event_type"])

def downgrade():
    op.drop_table("analytics_events")