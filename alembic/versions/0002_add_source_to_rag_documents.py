"""Agrega columna 'source' a rag_documents para clasificar el origen de cada documento.

Valores esperados:
  - "upload"                   → documento subido manualmente por el tenant
  - "agent_memory:user_explicit"   → hecho memorizado por el agente explícitamente
  - "agent_memory:preference"      → preferencia del usuario detectada por el agente
  - "agent_memory:{categoria}"     → cualquier categoría libre usada por store_tenant_memory

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-27 00:01:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Agrega 'source' a rag_documents.
    Idempotente via ADD COLUMN IF NOT EXISTS (Postgres 9.6+).
    Columna nullable para no bloquear filas existentes; default 'upload'.
    """
    op.execute("""
        ALTER TABLE rag_documents
        ADD COLUMN IF NOT EXISTS source VARCHAR(100) DEFAULT 'upload'
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_rag_documents_source
        ON rag_documents (source)
    """)

    # Backfill: filas existentes sin source quedan como 'upload'
    op.execute("""
        UPDATE rag_documents SET source = 'upload' WHERE source IS NULL
    """)


def downgrade() -> None:
    """Elimina la columna 'source' y su índice."""
    op.execute("DROP INDEX IF EXISTS ix_rag_documents_source")
    op.execute("ALTER TABLE rag_documents DROP COLUMN IF EXISTS source")
