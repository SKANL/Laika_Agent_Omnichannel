from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, JSON
from pgvector.sqlalchemy import Vector
from src.core.config import settings

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
