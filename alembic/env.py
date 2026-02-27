"""
Alembic Environment — Laika (Async SQLAlchemy + psycopg)
=========================================================
Configurado para ejecutar migraciones de forma SÍNCRONA via psycopg3
contra la misma base de datos PostgreSQL usada por la aplicación.

Notas de diseño:
- Las migraciones se ejecutan offline o con conexión directa (no asyncpg).
- env.py importa `settings` para obtener la URL real (nunca plain-text en alembic.ini).
- Los modelos SQLAlchemy de src.core.db se importan para que autogenerate
  detecte cambios de columnas automáticamente.
"""

import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ---------------------------------------------------------------------------
# Path fix: asegurarse de que 'src.*' sea importable desde este archivo.
# alembic/ está un nivel bajo del root del proyecto (Laika/).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Import settings ANTES de los modelos para que la BD esté disponible.
# ---------------------------------------------------------------------------
from src.core.config import settings

# ---------------------------------------------------------------------------
# Import de modelos para que autogenerate detecte cambios.
# IMPORTANTE: importar Base y todos los modelos que tienen tablas.
# ---------------------------------------------------------------------------
from src.core.db import Base
import src.core.db  # noqa: F401 — registra RAGDocument, SemanticCache en Base.metadata

# ---------------------------------------------------------------------------
# Alembic Config object — acceso a alembic.ini
# ---------------------------------------------------------------------------
config = context.config

# Sobreescribir el URL placeholder de alembic.ini con el URL real de settings.
# El proyecto usa psycopg3 (psycopg[binary]), NO psycopg2.
# SQLAlchemy requiere el prefijo `postgresql+psycopg://` para usar psycopg3.
# `psycopg_database_url` devuelve `postgresql://` (sin sufijo de driver),
# que SQLAlchemy 2.x resuelve como psycopg2 -> ModuleNotFoundError.
_alembic_url = settings.psycopg_database_url.replace(
    "postgresql://", "postgresql+psycopg://", 1
)
config.set_main_option("sqlalchemy.url", _alembic_url)

# Logging de Alembic desde la config del INI
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData objetivo para autogenerate
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# MIGRACIONES OFFLINE (sin conexión real, genera SQL puro)
# Útil para revisión de SQL antes de ejecutar en producción.
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Genera SQL de migraciones sin abrir una conexión real a la BD."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# MIGRACIONES ONLINE (conexión real a PostgreSQL)
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    """Ejecuta migraciones con conexión real a PostgreSQL via psycopg3."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # NullPool: una conexión por run, cierra al terminar
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,             # Detecta cambios de tipo de columna
            compare_server_default=True,   # Detecta cambios de valores default
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
