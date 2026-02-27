"""
task_dispatcher.py — Nodo de Despacho de Tareas en Background

Para intenciones clasificadas como 'tarea_larga', este nodo:
  1. Responde INMEDIATAMENTE al usuario con un mensaje de confirmación.
  2. Despacha la tarea pesada a Celery (fire-and-forget).
  3. Expone el task_id en state["background_task_id"] para trazabilidad.
  4. El grafo termina aquí (→ formatter → END) y el usuario no espera.

Cuando la tarea Celery termina, llama _dispatch_reply directamente
con el resultado definitivo. El usuario recibe DOS mensajes:
  - Primero: "Tu tarea ha sido iniciada..."
  - Después (minutos más tarde): el resultado real.

Ejemplos de intenciones de larga duración:
  - Generar un informe de ventas de los últimos 6 meses
  - Procesar y analizar un batch de 500 documentos
  - Ejecutar un pipeline de n8n que tarda varios minutos
"""
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.core.state import LaikaState
from structlog import get_logger

logger = get_logger("laika_task_dispatcher")


async def task_dispatcher_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Despacha una tarea de larga duración a Celery y responde al usuario de inmediato.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    thread_id = configurable.get("thread_id", "unknown")

    # Extraer el último mensaje del usuario como descripción de la tarea
    last_human_msg = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            last_human_msg = msg.content
            break

    # Despachar tarea pesada a Celery
    task_id = _dispatch_long_task(tenant_id, thread_id, last_human_msg)

    logger.info(
        "long_task_dispatched",
        task_id=task_id,
        tenant=tenant_id,
        thread=thread_id,
    )

    # Respuesta inmediata al usuario
    ack_message = (
        f"Tu solicitud ha sido registrada y está siendo procesada en segundo plano. "
        f"Recibirás la respuesta completa cuando el proceso termine.\n\n"
        f"Referencia de tarea: `{task_id}`\n"
        f"Puedes consultar el estado en /v1/jobs/{task_id}"
    )

    return {
        "messages": [AIMessage(content=ack_message)],
        "background_task_id": task_id,
    }


def _dispatch_long_task(tenant_id: str, thread_id: str, user_request: str) -> str:
    """
    Envía la tarea pesada a la cola Celery y retorna el task_id.
    Import diferido para evitar circular imports con celery_app.
    """
    from src.worker.tasks import run_long_background_task

    task = run_long_background_task.apply_async(
        kwargs={
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "user_request": user_request,
        }
    )
    return task.id
