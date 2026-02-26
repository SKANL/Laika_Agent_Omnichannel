from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

# Cargas Nucleares (Singleton settings e inicio de Logs)
from src.core.config import settings
from src.core.logging_setup import setup_logging
from src.api.routers import webhook

# Arrancar logs en formato JSON para que las consolas de Docker sean legibles
setup_logging(json_logs=True)
logger = get_logger("laika_main")

# ==========================================
# LAIKA V2: CEREBRO COGNITIVO B2B
# ==========================================
app = FastAPI(
    title="Laika Autonomous Agent Gateway",
    description="Motor de Inferencia Asíncrono Hibrido, Caching Semántico y LangGraph",
    version="2.0.0",
    docs_url="/docs", 
    redoc_url=None
)

# Hardening Base CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En Producción limitar a IP del N8N Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# INYECCIÓN DEL SISTEMA NERVIOSO (ROUTERS)
# ------------------------------------------
# Conectamos las rutas que escucharán a N8N y que dispararán al LLM
app.include_router(webhook.router)


@app.on_event("startup")
async def startup_event():
    """Validaciones ultra rápidas que FastAPI hace antes de abrir sus puertas."""
    logger.info("laika_gateway_starting", 
                version="2.0.0", 
                active_db=settings.POSTGRES_DB)
    
    # ==========================================
    # LANGFUSE v3: Inicializar Singleton Global
    # ==========================================
    # En Langfuse v3, se DEBE crear el singleton al inicio con Langfuse()
    # antes de usar cualquier CallbackHandler.
    # El SDK lee LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY y LANGFUSE_BASE_URL
    # automáticamente de las variables de entorno.
    try:
        from langfuse import Langfuse
        langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        logger.info("langfuse_singleton_initialized", host=settings.LANGFUSE_BASE_URL)
    except Exception as e:
        # Non-blocking: El agente funciona aunque Langfuse no esté disponible
        logger.warning("langfuse_singleton_init_failed", error=str(e))


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Latido rápido para orquestadores Docker/K8s."""
    return {"status": "Alive", "agent": "Laika V2 B2B"}
