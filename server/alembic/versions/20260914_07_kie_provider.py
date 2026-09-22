"""add the Kie model to the existing registry

Revision ID: 20260914_07
Revises: 20260820_06
"""
import uuid
from datetime import datetime, timezone
from alembic import op
import sqlalchemy as sa

revision = "20260914_07"
down_revision = "20260820_06"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    now = datetime.now(timezone.utc).isoformat()
    models = [
        ("gpt-6-astra", "GPT-6 Astra"), ("gpt-5.6", "GPT-5.6"),
        ("gpt-5.5", "GPT-5.5"), ("gpt-5.2", "GPT-5.2"),
        ("gemini-3.8-flash", "Gemini 3.8 Flash"),
        ("gemini-3.7-flash", "Gemini 3.7 Flash"),
        ("gemini-3.6-flash", "Gemini 3.6 Flash"),
        ("claude-sonnet", "Claude Sonnet"), ("claude-opus", "Claude Opus"),
        ("grok-4.6", "Grok 4.6 (via Kie)"), ("grok-4.5", "Grok 4.5 (via Kie)"),
        ("grok-4.3", "Grok 4.3 (via Kie)"), ("openai-codex", "OpenAI Codex (via Kie)"),
    ]
    for model_key, display_name in models:
        exists = conn.execute(sa.text("SELECT 1 FROM model_registry WHERE model_key = :key"), {"key": model_key}).scalar()
        if exists:
            continue
        conn.execute(sa.text("""
            INSERT INTO model_registry (id, provider, model_key, display_name, is_local, is_active, created_at, updated_at)
            VALUES (:id, 'kie', :model_key, :display_name, false, true, :now, :now)
        """), {"id": str(uuid.uuid4()), "model_key": model_key, "display_name": display_name, "now": now})


def downgrade():
    op.get_bind().execute(sa.text("DELETE FROM model_registry WHERE provider = 'kie'"))
