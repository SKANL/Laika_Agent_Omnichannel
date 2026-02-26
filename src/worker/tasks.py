from src.worker.celery_app import celery_app
from src.brain.workflows.main_graph import invoke_agent
from src.brain.tools.n8n_tool import trigger_dlq_webhook
from structlog import get_logger
import asyncio

logger = get_logger("laika_background_tasks")

# ==========================================
# RUTINAS ASÍNCRONAS EN SEGUNDO PLANO
# ==========================================

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # 1 minuto antes de reintentar tras Rate Limit
)
def process_agentic_workflow_celery(self, payload_dict: dict):
    """
    Consume el Payload JSON exacto que validó FastAPI (Pydantic) y ejecuta
    el grafo LangGraph de Laika de forma asíncrona.

    Notas de implementación:
    - asyncio.run() es el método correcto en Python 3.12+ para ejecutar coroutines
      desde contextos síncronos (Celery fork workers).
    - asyncio.get_event_loop() está deprecado en Python 3.12 en hilos secundarios
      (como los workers de Celery) y lanza DeprecationWarning o RuntimeError.
    - asyncio.run() crea un event loop nuevo y lo cierra limpiamente al terminar,
      lo que evita leaks de conexiones entre tareas del mismo worker.
    """
    tenant_id = payload_dict.get("tenant_id")
    thread_id = payload_dict.get("thread_id")
    user_query = payload_dict.get("user_query")

    logger.info("celery_processing_start", tenant=tenant_id, thread=thread_id)

    try:
        # asyncio.run() crea un nuevo event loop, ejecuta la coroutine y lo cierra.
        # Es el único patrón correcto y seguro en Python 3.10+ para Celery workers.
        channel = payload_dict.get("channel", "unknown")
        extra_metadata = payload_dict.get("metadata") or {}
        asyncio.run(invoke_agent(tenant_id, thread_id, user_query, channel=channel, extra_metadata=extra_metadata))

        logger.info("celery_processing_success", tenant=tenant_id)
        return {"status": "success"}

    except Exception as exc:
        logger.error("celery_processing_failed", error=str(exc))
        try:
            # Reintenta si el error parece transitorio (Rate Limit, timeout, etc.)
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.critical("max_retries_exceeded", tenant=tenant_id)
            # DLQ: alerta HTTP de emergencia a n8n para no silenciar la caída.
            asyncio.run(
                trigger_dlq_webhook(
                    tenant_id,
                    thread_id,
                    f"Error final en Grafo Laika tras {self.max_retries} reintentos: {str(exc)}",
                )
            )
            raise


# ------------------------------------------
# CRON JOB: AUTONOMIA PROACTIVA (HEARTBEAT)
# ------------------------------------------

@celery_app.task
def proactive_heartbeat_trigger(reason: str):
    """
    Llamada automaticamente por Celery Beat segun el beat_schedule configurado.
    En produccion: iterar sobre tenants activos y disparar invoke_agent por cada uno.
    """
    logger.info("proactive_heartbeat_fired", reason=reason)
    # TODO Fase 2: iterar sobre tenant configs activos en Postgres y despertar agentes.
    return {"status": "heartbeat_acknowledged"}

