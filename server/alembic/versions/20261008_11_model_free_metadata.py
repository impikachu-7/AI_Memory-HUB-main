"""Add provider-reported free access metadata to model registry.

Revision ID: 20261008_11
Revises: 20260922_10
"""
from alembic import op
import sqlalchemy as sa

revision = "20261008_11"
down_revision = "20260922_10"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("model_registry", sa.Column("is_free", sa.Boolean(), nullable=True))
    # The original unnamed unique constraint reflects with a generated name on
    # Postgres and no name on SQLite; batch naming makes both schemas migratable.
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("model_registry", naming_convention={"uq": "uq_%(table_name)s_%(column_0_name)s"}) as batch:
            batch.drop_constraint("uq_model_registry_model_key", type_="unique")
            batch.create_unique_constraint("uq_model_registry_provider_model", ["provider", "model_key"])
    else:
        with op.batch_alter_table("model_registry") as batch:
            batch.drop_constraint("model_registry_model_key_key", type_="unique")
            batch.create_unique_constraint("uq_model_registry_provider_model", ["provider", "model_key"])


def downgrade():
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("model_registry", naming_convention={"uq": "uq_%(table_name)s_%(column_0_name)s"}) as batch:
            batch.drop_constraint("uq_model_registry_provider_model", type_="unique")
            batch.create_unique_constraint("uq_model_registry_model_key", ["model_key"])
    else:
        with op.batch_alter_table("model_registry") as batch:
            batch.drop_constraint("uq_model_registry_provider_model", type_="unique")
            batch.create_unique_constraint("model_registry_model_key_key", ["model_key"])
    op.drop_column("model_registry", "is_free")
