"""Baseline inicial — captura el esquema existente creado por init_db()

Esta migración representa el estado inicial del esquema para proyectos que
ya tienen las tablas creadas por SQLAlchemy (init_db / Base.metadata.create_all).

Para una instalación EXISTENTE (tablas ya creadas por init_db):
    alembic stamp 001_baseline
    alembic upgrade head

Para una instalación NUEVA (BD en blanco):
    alembic upgrade head          # aplica baseline + migraciones posteriores

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-02-27 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Crea las tablas base si no existen.
    Idempotente: usa IF NOT EXISTS para no fallar en instalaciones existentes.
    """
    # Activar extensión pgvector (idempotente)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Tabla de documentos RAG (sin columna 'source' — la agrega la migración 0002)
    op.execute("""
        CREATE TABLE IF NOT EXISTS rag_documents (
            id          SERIAL PRIMARY KEY,
            tenant_id   VARCHAR(50)  NOT NULL,
            content     TEXT         NOT NULL,
            embedding   vector(384),
            metadata_json JSONB      DEFAULT '{}'
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_rag_documents_tenant_id
        ON rag_documents (tenant_id)
    """)

    # Tabla de caché semántico
    op.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id                  SERIAL PRIMARY KEY,
            tenant_id           VARCHAR(50)  NOT NULL,
            question            TEXT         NOT NULL,
            question_embedding  vector(384),
            pre_computed_answer TEXT         NOT NULL
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_semantic_cache_tenant_id
        ON semantic_cache (tenant_id)
    """)


def downgrade() -> None:
    """Elimina las tablas base. CUIDADO: destruye datos en producción."""
    op.drop_table("semantic_cache")
    op.drop_table("rag_documents")
