from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from src.core.state import LaikaState
from src.brain.llm_proxy import get_orchestrator_llm
from src.brain.rate_limiter import set_model_cooldown, record_model_usage
from src.brain.tools.rag_tool import perform_rag_search
from src.brain.tools.n8n_tool import n8n_workflow_execution
from src.brain.tools.web_search_tool import web_search
from structlog import get_logger

logger = get_logger("laika_orchestrator")

_MAX_MODEL_ROTATIONS = 3  # intentos máximos cambiando de modelo ante un 429

# ==========================================
# HERRAMIENTAS DISPONIBLES PARA EL ORQUESTADOR
# ==========================================
# El LLM puede invocar cualquiera de estas herramientas de forma autonoma.
# LangGraph maneja el loop tool-call -> tool_result -> LLM de forma nativa.
ORCHESTRATOR_TOOLS = [perform_rag_search, n8n_workflow_execution, web_search]

# ToolNode ejecuta los tool_calls que el LLM emite y devuelve ToolMessages al State.
orchestrator_tool_node = ToolNode(ORCHESTRATOR_TOOLS)


async def orchestrator_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patron 2: Orchestrator-Worker con Agentic RAG.

    Este nodo usa el Heavy Lifter (70B) con bind_tools().
    El LLM decide autonomamente si invocar:
      - perform_rag_search: busca en la memoria documental del tenant
      - web_search: busca informacion publica en tiempo real via Tavily
      - n8n_workflow_execution: ordena a n8n ejecutar una accion en el mundo real

    El grafo redirige al orchestrator_tool_node si el LLM emite tool_calls,
    y vuelve a este nodo con los ToolMessages hasta que el LLM responda sin tools.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    intent = state.get("current_intent", "unknown")
    retry = state.get("retry_count", 0)

    logger.info("orchestrator_node_start", intent=intent, tenant=tenant_id, retry=retry)

    # Prompt de sistema con contexto dinámico (Prompt Injection mínima)
    # (La selección del LLM ahora ocurre dentro del loop de rotación más abajo)
    context_instruction = (
        f"Atiende al usuario del Tenant '{tenant_id}'. "
        f"Intencion clasificada: '{intent}'. "
        "Usa las herramientas disponibles para buscar informacion antes de responder. "
        "Prioriza perform_rag_search para conocimiento interno de la empresa, "
        "y web_search solo para datos publicos o en tiempo real que el RAG no tiene."
    )
    if retry > 0:
        context_instruction += (
            f" REINTENTO {retry}: El evaluador rechazo el borrador anterior. "
            "Mejora la respuesta corrigiendo los problemas indicados."
        )

    system_msg = SystemMessage(content=context_instruction)

    # CRITICO: pasar `config` propaga los callbacks de Langfuse al LLM.
    messages_to_send = [system_msg] + state["messages"]
    # Cap de seguridad: evitar context overflow con hilos largos
    if len(messages_to_send) > 25:
        messages_to_send = [system_msg] + state["messages"][-20:]

    # ── ROTACIÓN AUTOMÁTICA EN 429 ────────────────────────────────
    # Si el modelo seleccionado devuelve 429, se pone en cooldown y
    # se reintenta con el siguiente mejor modelo del pool automáticamente.
    response = None
    for _attempt in range(_MAX_MODEL_ROTATIONS):
        llm = await get_orchestrator_llm()
        llm_with_tools = llm.bind_tools(ORCHESTRATOR_TOOLS)
        model_id = getattr(llm, "_laika_model_id", "unknown")
        try:
            response = await llm_with_tools.ainvoke(messages_to_send, config=config)
            # Registrar uso real (RPM; tokens no disponibles sin metadata del response)
            await record_model_usage(model_id, 0)
            break
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                logger.warning("orchestrator_429_cooldown",
                               model=model_id, attempt=_attempt + 1)
                await set_model_cooldown(model_id, seconds=60)
                if _attempt < _MAX_MODEL_ROTATIONS - 1:
                    continue
            raise

    if response is None:
        raise RuntimeError("Todos los modelos de orchestration agotados")

    logger.info("orchestrator_node_completed",
                tenant=tenant_id,
                has_tool_calls=bool(getattr(response, "tool_calls", None)))

    return {"messages": [response]}


def should_use_tools(state: LaikaState) -> str:
    """
    Borde condicional post-orchestrator.
    Si el LLM emitio tool_calls -> ejecutar herramientas -> volver al orquestador.
    Si no -> pasar al evaluador.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    return "evaluate"

