"""add configurable model capability fields

Revision ID: 20260922_10
Revises: 20260922_09
"""
from alembic import op
import sqlalchemy as sa

revision = "20260922_10"
down_revision = "20260922_09"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("model_registry", sa.Column("context_window", sa.Integer(), nullable=True))
    op.add_column("model_registry", sa.Column("max_output_tokens", sa.Integer(), nullable=True))
    op.add_column("model_registry", sa.Column("supports_streaming", sa.Boolean(), nullable=True))
    op.add_column("model_registry", sa.Column("supports_temperature", sa.Boolean(), nullable=True))

def downgrade():
    op.drop_column("model_registry", "supports_temperature")
    op.drop_column("model_registry", "supports_streaming")
    op.drop_column("model_registry", "max_output_tokens")
    op.drop_column("model_registry", "context_window")