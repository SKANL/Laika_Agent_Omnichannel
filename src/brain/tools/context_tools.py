"""
context_tools.py — Herramientas de Gestión de Contexto y Memoria

Implementa los patrones de Context Engineering para prevenir:
  - Context Rot (degradación por ruido acumulado)
  - Lost in the Middle (información importante enterrada en mensajes intermedios)
  - Context Window overflow (exceder los límites físicos del modelo)

Herramientas incluidas:
  - summarize_conversation: Compacta el historial de mensajes en un resumen
    estructurado, liberando espacio en el context window sin perder información clave.
  - check_task_status: Consulta el estado de tareas Celery de larga duración.
  - store_tenant_memory: Persiste un hecho explícito en la memoria RAG del tenant.

Referencias:
  - Context Compaction Pattern (Claude Code docs, Anthropic 2024)
  - Checkpoint & Reset Pattern (Karpathy "Context Engineering", 2025)
  - Reflexion (Shinn et al. 2023) — self-critique y acumulación de memoria
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from structlog import get_logger

logger = get_logger("laika_context_tools")


# ============================================================================
# HERRAMIENTA 4: COMPACTACIÓN DE CONTEXTO
# ============================================================================

@tool
async def summarize_conversation(
    messages_json: str,
    config: RunnableConfig = None,
) -> str:
    """
    Compacta el historial de una conversación larga en un resumen estructurado.
    Usa esta herramienta cuando el historial supere los 15 mensajes o cuando
    notes que la conversación se está volviendo repetitiva.

    El resumen preserva: decisiones tomadas, entidades mencionadas, contexto
    clave, tareas pendientes. Elimina: saludos, mensajes redundantes, respuestas
    de confirmación simples.

    USA ESTA HERRAMIENTA para "resetear" el contexto sin perder información
    crítica — implementa el patrón "Context Compaction" documentado por Anthropic.

    Args:
        messages_json: Lista de mensajes en formato JSON string.
                       Formato: [{"role": "user"|"assistant", "content": "texto"}, ...]

    Returns:
        Resumen compacto en formato string listo para insertar como contexto.

    Ejemplo:
        summarize_conversation('[{"role": "user", "content": "..."}, ...]')
        → "RESUMEN PREVIO DE CONVERSACIÓN:\n- Usuario consultó sobre contrato ABC (orden #1234)..."
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.brain.llm_proxy import get_routing_llm

    logger.info("summarize_conversation_called")

    try:
        messages = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
    except json.JSONDecodeError:
        return "Error: messages_json no es JSON válido."

    if not messages:
        return "No hay mensajes para resumir."

    # Formatear los mensajes para el LLM
    convo_text = "\n".join([
        f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:500]}"
        for m in messages[-30:]  # Máximo 30 mensajes recientes
    ])

    sys_msg = SystemMessage(
        content=(
            "Eres un compactador de conversaciones. Resume la conversación preservando:\n"
            "1. Decisiones y acuerdos alcanzados\n"
            "2. Entidades mencionadas (IDs, montos, fechas, nombres)\n"
            "3. Tareas solicitadas y su estado (pendiente/completada)\n"
            "4. El contexto de la consulta principal\n\n"
            "Elimina: saludos, confirmaciones simples, información redundante.\n"
            "Formato: viñetas concisas. Máximo 300 palabras.\n"
            "Empieza con: 'RESUMEN PREVIO DE CONVERSACIÓN:'"
        )
    )
    user_msg = HumanMessage(content=f"Resume esta conversación:\n\n{convo_text}")

    try:
        llm = await get_routing_llm()
        response = await llm.ainvoke([sys_msg, user_msg], config=config)
        summary = response.content.strip()
        logger.info("summarize_conversation_success", chars=len(summary))
        return summary
    except Exception as e:
        logger.error("summarize_conversation_failed", error=str(e))
        return f"Error al resumir conversación: {e}"


# ============================================================================
# HERRAMIENTA 5: CONSULTA DE ESTADO DE TAREA CELERY
# ============================================================================

@tool
def check_task_status(task_id: str, config: RunnableConfig = None) -> str:
    """
    Consulta el estado actual de una tarea en segundo plano (Celery).
    USA ESTA HERRAMIENTA cuando el usuario pregunte por el estado de una
    tarea larga que fue enviada previamente (referencia: tarea_larga).

    Args:
        task_id: El ID de la tarea Celery. Formato UUID.
                 Ejemplo: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    Returns:
        Estado de la tarea y resultado parcial si está disponible.

    Estados posibles: PENDING, STARTED, SUCCESS, FAILURE, RETRY
    """
    logger.info("check_task_status_called", task_id=task_id[:36])

    try:
        from src.worker.celery_app import celery_app
        from celery.result import AsyncResult  # type: ignore[import-untyped]

        result = AsyncResult(task_id, app=celery_app)
        state = result.state

        if state == "PENDING":
            return f"Tarea '{task_id}': en cola, aún no iniciada."
        elif state == "STARTED":
            return f"Tarea '{task_id}': en procesamiento actualmente."
        elif state == "SUCCESS":
            task_result = result.result
            if isinstance(task_result, dict):
                summary = task_result.get("summary", str(task_result)[:200])
            else:
                summary = str(task_result)[:200]
            return f"Tarea '{task_id}': COMPLETADA. Resultado: {summary}"
        elif state == "FAILURE":
            error_info = str(result.result)[:200] if result.result else "Error desconocido"
            return f"Tarea '{task_id}': FALLÓ. Error: {error_info}"
        elif state == "RETRY":
            return f"Tarea '{task_id}': reintentando tras error temporal."
        else:
            return f"Tarea '{task_id}': estado '{state}'."

    except Exception as e:
        logger.error("check_task_status_failed", task_id=task_id, error=str(e))
        return f"Error al consultar el estado de la tarea '{task_id}': {e}"


# ============================================================================
# HERRAMIENTA 6: ALMACENAR MEMORIA EXPLÍCITA DEL TENANT
# ============================================================================

@tool
async def store_tenant_memory(
    fact: str,
    category: str = "user_explicit",
    config: RunnableConfig = None,
) -> str:
    """
    Persiste un hecho explícito en la base de conocimientos del tenant.
    Implementa el patrón de Memoria Conversacional Acumulativa (Reflexion, Shinn et al. 2023).

    USA ESTA HERRAMIENTA cuando el usuario proporcione información que debe recordarse
    en futuras conversaciones: preferencias, configuraciones, datos de negocio importantes.
    NUNCA almacenes información sensible (contraseñas, tokens, datos PII sin consentimiento).

    Args:
        fact: El hecho a almacenar. Sé específico y conciso.
              Malo:  "el usuario dijo algo sobre precios"
              Bueno: "El cliente prefiere comunicación por correo. Contacto: gerencia@empresa.com"
        category: Categoría para organizar la memoria.
                  Valores sugeridos: "user_explicit", "preference", "business_rule", "contact"

    Returns:
        Confirmación del almacenamiento o error.
    """
    from src.brain.embeddings import encode_text
    from src.core.db import AsyncSessionLocal, RAGDocument

    configurable = (config or {}).get("configurable", {}) if config else {}
    tenant_id = configurable.get("tenant_id", "")

    if not tenant_id:
        return "Error: tenant_id no disponible. No se puede persistir la memoria."

    if not fact or len(fact.strip()) < 10:
        return "Error: El hecho a almacenar es demasiado corto. Proporciona más contexto."

    logger.info("store_tenant_memory_called", tenant_id=tenant_id, category=category)

    try:
        embedding = encode_text(fact)
        content_with_meta = f"[{category.upper()}] {fact}"

        async with AsyncSessionLocal() as session:
            doc = RAGDocument(
                tenant_id=tenant_id,
                content=content_with_meta,
                embedding=embedding,
                source=f"agent_memory:{category}",
            )
            session.add(doc)
            await session.commit()

        logger.info("store_tenant_memory_success", tenant_id=tenant_id)
        return f"Memoria almacenada correctamente en la base de conocimientos de '{tenant_id}'."

    except Exception as e:
        logger.error("store_tenant_memory_failed", error=str(e), tenant_id=tenant_id)
        return f"Error al almacenar la memoria: {e}"
