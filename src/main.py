from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

# Cargas Nucleares (Singleton settings e inicio de Logs)
from src.core.config import settings
from src.core.db import init_db
from src.core.logging_setup import setup_logging
from src.api.routers import webhook, documents, health, jobs, tenants

# Arrancar logs en formato JSON para que las consolas de Docker sean legibles
setup_logging(json_logs=True)
logger = get_logger("laika_main")


# ==========================================
# LIFESPAN (reemplaza @app.on_event deprecated)
# Patrón recomendado en FastAPI 0.93+ / Starlette 0.27+
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: inicializa singletons costosos antes de servir peticiones.
    Shutdown: flush de observabilidad y cierre limpio de pools.
    """

    # --- STARTUP ---
    logger.info("laika_gateway_starting", version="2.0.0", active_db=settings.POSTGRES_DB)

    # 0. Inicializar tablas SQLAlchemy + extensión pgvector
    # Importar modelos para que SQLAlchemy los registre con Base.metadata
    import src.core.tenant_config  # noqa: F401 — registra TenantConfig en Base
    try:
        await init_db()
    except Exception as e:
        logger.error("db_init_failed", error=str(e))
        raise  # Crítico: no podemos arrancar sin DB

    # 1. Langfuse v3: registrar singleton global para que CallbackHandler lo reutilice
    try:
        from langfuse import Langfuse
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        logger.info("langfuse_singleton_initialized", host=settings.LANGFUSE_BASE_URL)
    except Exception as e:
        # Non-blocking: el agente funciona sin Langfuse en entornos sin acceso externo
        logger.warning("langfuse_singleton_init_failed", error=str(e))

    # 2. Pre-cargar el modelo de embeddings para que el primer request no pague el cold start
    try:
        from src.brain.embeddings import get_embedding_model
        model = get_embedding_model()
        logger.info("embedding_model_preloaded", device=str(model.device))
    except Exception as e:
        logger.warning("embedding_model_preload_failed", error=str(e))

    yield  # <-- la app sirve requests entre yield y el bloque de shutdown

    # --- SHUTDOWN ---
    try:
        from langfuse import Langfuse
        Langfuse().flush()
        logger.info("langfuse_flushed_on_shutdown")
    except Exception:
        pass

    logger.info("laika_gateway_shutdown_complete")


# ==========================================
# LAIKA V2: CEREBRO COGNITIVO B2B
# ==========================================
app = FastAPI(
    title="Laika Autonomous Agent Gateway",
    description="Motor de Inferencia Asíncrono Híbrido, Caching Semántico y LangGraph",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# Hardening Base CORS — orígenes configurados via settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# INYECCIÓN DEL SISTEMA NERVIOSO (ROUTERS)
# ------------------------------------------
app.include_router(webhook.router)
app.include_router(documents.router)
app.include_router(health.router)   # GET /health, /health/models, /health/rotation
app.include_router(jobs.router)     # GET /v1/jobs/{task_id} — estado de tareas Celery
app.include_router(tenants.router)  # CRUD /v1/tenants — configuración por tenant
