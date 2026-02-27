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
    Disparado automáticamente por Celery Beat segun el beat_schedule.
    
    Implementacion:
    1. Consulta rag_documents para obtener los tenant_ids unicos activos.
       (Proxy de tenants configurados: si han subido documentos, están activos).
    2. Por cada tenant, invoca el agente con un mensaje de contexto interno
       que puede usar para tareas proactivas (resumen diario, alertas, etc.).
    """
    logger.info("proactive_heartbeat_fired", reason=reason)
    
    async def _run_heartbeats():
        from sqlalchemy import select, distinct
        from src.core.db import AsyncSessionLocal, RAGDocument

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(distinct(RAGDocument.tenant_id))
                )
                tenant_ids = [row[0] for row in result.all()]
        except Exception as e:
            logger.error("heartbeat_tenant_query_failed", error=str(e))
            return {"status": "error", "detail": str(e)}

        if not tenant_ids:
            logger.info("heartbeat_no_tenants_found")
            return {"status": "no_tenants"}

        logger.info("heartbeat_tenants_found", count=len(tenant_ids), tenants=tenant_ids)

        errors = []
        for tenant_id in tenant_ids:
            try:
                await invoke_agent(
                    tenant_id=tenant_id,
                    thread_id=f"heartbeat_{reason}",
                    payload_msg=(
                        f"[SISTEMA] Pulso proactivo matutino ({reason}). "
                        "Si hay tareas pendientes, resumen o alertas configuradas para hoy, "
                        "precéralas y despacha al canal correspondiente."
                    ),
                    channel="heartbeat",
                    extra_metadata={"heartbeat_reason": reason},
                )
                logger.info("heartbeat_agent_invoked", tenant=tenant_id)
            except Exception as exc:
                logger.error("heartbeat_agent_failed", tenant=tenant_id, error=str(exc))
                errors.append({"tenant": tenant_id, "error": str(exc)})

        return {
            "status": "completed",
            "tenants_processed": len(tenant_ids),
            "errors": errors,
        }

    result = asyncio.run(_run_heartbeats())
    logger.info("proactive_heartbeat_complete", result=result)
    return result


# ------------------------------------------
# TAREA DE LARGA DURACIÓN CON NOTIFICACIÓN
# ------------------------------------------

@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=120,  # 2 minutos antes de reintentar
    time_limit=1800,           # 30 minutos máximo por tarea larga
    soft_time_limit=1500,      # SoftTimeLimitExceeded a los 25 min (para cleanup)
)
def run_long_background_task(self, tenant_id: str, thread_id: str, user_request: str):
    """
    Ejecuta una tarea de larga duración en background y notifica al usuario cuando termina.

    Ciclo de vida:
      1. task_dispatcher_node kick-off: el usuario ya recibió el ACK inmediato.
      2. Esta tarea corre en Celery con time_limit=30min.
      3. Al finalizar con éxito, llama _dispatch_reply con el resultado completo.
      4. Si falla después de max_retries, envía notificación de error al usuario.

    El thread_id tiene sufijo "_bg" para separar el historial del hilo background
    del hilo principal (no queremos que el orquestador "recuerde" el ACK como contexto).
    """
    # Thread separado para el trabajo background
    bg_thread_id = f"{thread_id}_bg"

    logger.info(
        "long_task_start",
        task_id=self.request.id,
        tenant=tenant_id,
        bg_thread=bg_thread_id,
    )

    async def _run():
        from src.brain.workflows.main_graph import invoke_agent, _dispatch_reply
        try:
            await invoke_agent(
                tenant_id=tenant_id,
                thread_id=bg_thread_id,        # checkpointer usa namespace aislado
                payload_msg=user_request,
                channel="background",
                extra_metadata={
                    "task_type": "long_background",
                    "original_thread": thread_id,
                    "celery_task_id": self.request.id,
                },
                reply_thread_id=thread_id,     # respuesta → thread ORIGINAL del usuario
            )
            logger.info("long_task_invoke_complete", tenant=tenant_id)
        except Exception as exc:
            # Si el agente falla, notificar al usuario en lugar de dejar silencio
            error_msg = (
                f"Lo siento, tu tarea en segundo plano no pudo completarse.\n"
                f"Error: {type(exc).__name__}\n"
                f"Referencia: {self.request.id}\n\n"
                "Por favor intenta de nuevo o contacta al administrador."
            )
            await _dispatch_reply(tenant_id, thread_id, error_msg)
            raise

    try:
        asyncio.run(_run())
        logger.info("long_task_success", task_id=self.request.id, tenant=tenant_id)
        return {"status": "success", "task_id": self.request.id}

    except Exception as exc:
        logger.error("long_task_failed", error=str(exc), task_id=self.request.id)
        try:
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.critical("long_task_max_retries_exceeded", tenant=tenant_id)
            # Notificación de fallo final (ya enviada dentro de _run() al usuario)
            raise
