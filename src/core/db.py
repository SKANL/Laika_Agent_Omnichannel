from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Boolean, JSON, DateTime, text, func
from pgvector.sqlalchemy import Vector
from src.core.config import settings
import structlog

_db_logger = structlog.get_logger("laika_db")

# ==========================================
# GESTOR ASÍNCRONO DE POSTGRES & PGVECTOR
# ==========================================
# FastAPI es Asíncrono. Si usamos conectores síncronos, 
# cada llamada de vector search ahogaría todo el servidor.

# El engine se conecta con asyncpg (no psycopg2).
engine = create_async_engine(
    settings.async_database_url,
    pool_size=10,
    max_overflow=20,
    echo=False # Útil en dev, loguea las sentencias SQL reales
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# ==========================================
# ESQUEMAS: EL ESCUDO MULTITENANT
# ==========================================

class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 🔴 SEGURIDAD B2B: Esta columna nunca puede estar vacía.
    # Divide físicamente los clientes en la misma base de datos.
    tenant_id = Column(String(50), nullable=False, index=True)

    content = Column(String, nullable=False)
    # Vector de embebimientos (ej. paraphrase-multilingual-MiniLM-L12-v2 da 384 dims)
    embedding = Column(Vector(384))

    # Origen del documento: "upload", "agent_memory:user_explicit", "agent_memory:preference", etc.
    # Permite filtrar por tipo de memoria en búsquedas RAG.
    source = Column(String(100), nullable=True, index=True, default="upload")

    metadata_json = Column(JSON, default={})

class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True) 
    
    question = Column(String, nullable=False)
    question_embedding = Column(Vector(384))
    
    pre_computed_answer = Column(String, nullable=False)

# Función inyectable para FastAPI `Depends()`
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """
    Inicializa la base de datos al arrancar la aplicación:
      1. Activa la extensión pgvector (CREATE EXTENSION IF NOT EXISTS vector).
      2. Crea todas las tablas SQLAlchemy con CREATE TABLE IF NOT EXISTS.

    El LangGraph checkpointer (AsyncPostgresSaver.setup()) se llama por separado
    en main_graph.invoke_agent, ya que requiere un pool psycopg nativo.

    Esta función es idempotente: puede llamarse múltiples veces sin problema.
    """
    async with engine.begin() as conn:
        # Activar extensión pgvector: necesaria para el tipo vector(384)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Crear tablas SQLAlchemy (rag_documents, semantic_cache)
        await conn.run_sync(Base.metadata.create_all)
    _db_logger.info(
        "db_initialized",
        tables=[t.name for t in Base.metadata.sorted_tables],
    )
