from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init
from src.core.config import settings
from structlog import get_logger

logger = get_logger("laika_celery_worker")

# ------------------------------------------
# POST-FORK: RESET DEL MODELO DE EMBEDDINGS
# ------------------------------------------
# En Linux, Celery usa prefork workers (fork()). El proceso hijo hereda
# el estado del padre incluyendo el contexto CUDA del modelo de embeddings.
# Esto causa errores de CUDA en workers concurrentes con GPU.
# Solucion: resetear la instancia del modelo en cada worker hijo antes
# de que procese cualquier tarea, forzando re-inicializacion limpia.
@worker_process_init.connect
def reset_embedding_model_on_fork(**kwargs):
    """
    Restablece el modelo de embeddings tras fork() de Celery worker.
    Solo relevante en Linux/macOS con modelos GPU (CUDA/MPS).
    En Windows (spawn, no fork) esta signal igual se dispara pero es inocuo.
    """
    try:
        import src.brain.embeddings as _emb
        _emb._model_instance = None
        logger.info("embedding_model_reset_on_fork")
    except Exception as e:
        logger.warning("embedding_model_reset_failed", error=str(e))


# ==========================================
# GESTOR DE TRABAJOS EN SEGUNDO PLANO (REDIS)
# ==========================================

# Conectamos con el broker que levantamos en docker-compose
celery_app = Celery(
    "laika_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["src.worker.tasks"] # Archivo donde empaquetaremos los jobs
)

# ==========================================
# LANGFUSE v3: Inicializar Singleton Global para el Worker
# ==========================================
# El Celery Worker es un proceso independiente del FastAPI App.
# Necesita su propio singleton de Langfuse para poder trazar.
try:
    from langfuse import Langfuse
    _langfuse = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_BASE_URL,
    )
    logger.info("celery_langfuse_initialized", host=settings.LANGFUSE_BASE_URL)
except Exception as e:
    logger.warning("celery_langfuse_init_failed", error=str(e))

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.TIMEZONE,
    enable_utc=True,
    task_track_started=True,
    # 🔴 Resiliencia: Configuración de Límites y Retries
    task_publish_retry=True,
    task_publish_retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    }
)

# ------------------------------------------
# LATIDO PROACTIVO (CRON)
# ------------------------------------------
celery_app.conf.beat_schedule = {
    # Un pulso que despierta al agente a las 8:00 AM todos los días.
    "morning_heartbeat_check": {
        "task": "src.worker.tasks.proactive_heartbeat_trigger",
        # Schedule cron de Linux
        "schedule": crontab(hour=8, minute=0), 
        "args": ("heartbeat_morning",)
    }
}
