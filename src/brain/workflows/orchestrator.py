import os
import asyncio
import yaml
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from src.core.state import LaikaState
from src.brain.llm_proxy import get_orchestrator_llm
from src.brain.rate_limiter import set_model_cooldown, record_model_usage
from src.brain.tools.rag_tool import perform_rag_search
from src.brain.tools.n8n_tool import n8n_workflow_execution
from src.brain.tools.web_search_tool import web_search
from src.brain.tools.deterministic_tools import calculate, get_current_datetime, extract_entities
from src.brain.tools.context_tools import summarize_conversation, check_task_status, store_tenant_memory
from structlog import get_logger

logger = get_logger("laika_orchestrator")

_MAX_MODEL_ROTATIONS = 3

# Path absoluto para compatibilidad con uvicorn y celery worker
_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "prompts_registry.yaml")
with open(_PROMPTS_PATH, "r", encoding="utf-8") as _f:
    _prompts = yaml.safe_load(_f)

# ==========================================
# HERRAMIENTAS DISPONIBLES PARA EL ORQUESTADOR
# ==========================================
# El LLM puede invocar cualquiera de estas herramientas de forma autonoma.
# LangGraph maneja el loop tool-call -> tool_result -> LLM de forma nativa.
#
# Grupos:
#   Cognitivas:     perform_rag_search, web_search, n8n_workflow_execution
#   El Escudo:      calculate, get_current_datetime, extract_entities
#   Contexto/Mem:   summarize_conversation, check_task_status, store_tenant_memory
ORCHESTRATOR_TOOLS = [
    # --- Cognitivas (LLM-assisted) ---
    perform_rag_search,
    web_search,
    n8n_workflow_execution,
    # --- El Escudo (Deterministas — costo $0, sin riesgo de alucinación) ---
    calculate,
    get_current_datetime,
    extract_entities,
    # --- Contexto y Memoria ---
    summarize_conversation,
    check_task_status,
    store_tenant_memory,
]

# ToolNode ejecuta los tool_calls que el LLM emite y devuelve ToolMessages al State.
orchestrator_tool_node = ToolNode(ORCHESTRATOR_TOOLS)


async def orchestrator_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patron 2: Orchestrator-Worker con Agentic RAG.

    Para intenciones 'investigacion_complex' con plan disponible:
      - Corre RAG + web_search en PARALELO via asyncio.gather como pre-fetch.
      - Inyecta los resultados como contexto adicional al LLM antes del tool loop.

    Para el resto de intenciones:
      - LLM con bind_tools decide autonomamente cuando invocar herramientas.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    intent = state.get("current_intent", "unknown")
    retry = state.get("retry_count", 0)
    plan = state.get("plan", [])

    logger.info("orchestrator_node_start", intent=intent, tenant=tenant_id, retry=retry)

    # --- System prompt con backstory + prompt del YAML ---
    backstory = _prompts.get("global_backstory", "")
    orchestrator_prompt = _prompts.get("system_prompts", {}).get(
        "orchestrator_node",
        "Eres Laika, el orquestador cognitivo. Responde con precision usando las herramientas disponibles.",
    )

    context_parts = [f"{backstory}\n\n{orchestrator_prompt}"]
    context_parts.append(f"\nTenant activo: '{tenant_id}'. Intencion clasificada: '{intent}'.")

    if plan:
        plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
        context_parts.append(f"\nPLAN DE EJECUCION (sigue estos pasos en orden):\n{plan_text}")

    if retry > 0:
        context_parts.append(
            f"\n⚠ REINTENTO {retry}: El evaluador rechazo el borrador anterior. "
            "Corrige los problemas indicados en la critica interna."
        )

    # --- Pre-fetch paralelo para investigacion_complex ---
    pre_context = ""
    if intent == "investigacion_complex" and plan:
        logger.info("orchestrator_parallel_prefetch_start", tenant=tenant_id, steps=len(plan))
        first_query = plan[0] if plan else state["messages"][-1].content

        async def _safe_rag(q: str) -> str:
            try:
                return await perform_rag_search.ainvoke({"query": q}, config=config)
            except Exception as e:
                return f"[RAG error: {e}]"

        async def _safe_web(q: str) -> str:
            try:
                return await web_search.ainvoke({"query": q}, config=config)
            except Exception as e:
                return f"[Web error: {e}]"

        rag_result, web_result = await asyncio.gather(
            _safe_rag(first_query),
            _safe_web(first_query),
        )
        pre_context = (
            f"\n\n<pre_fetched_context>"
            f"\n<rag_results>\n{rag_result}\n</rag_results>"
            f"\n<web_results>\n{web_result}\n</web_results>"
            f"\n</pre_fetched_context>"
        )
        context_parts.append(pre_context)
        logger.info("orchestrator_parallel_prefetch_done", tenant=tenant_id)

    system_msg = SystemMessage(content="\n".join(context_parts))

    # ── CONTEXT TRIMMING INTELIGENTE ──────────────────────────────────────────
    # Mitigación "Lost in the Middle" (Liu et al. 2023):
    # - El mensaje más antiguo del usuario (consulta original) se ANCLA al inicio.
    #   Garantiza que el LLM no "olvide" el objetivo original en conversaciones largas.
    # - Los mensajes intermedios se recortan cuando hay demasiados.
    # - Los mensajes más recientes siempre se preservan (sesgo de recencia).
    all_messages = state["messages"]
    _CONTEXT_CAP = 24  # Máximo mensajes en el context (excl. system_msg)
    if len(all_messages) > _CONTEXT_CAP:
        # Pin: primer HumanMessage (consulta original) + últimos N mensajes
        first_human = next(
            (m for m in all_messages if getattr(m, "type", "") == "human"),
            None,
        )
        recent = all_messages[-(_CONTEXT_CAP - 1):]
        if first_human and first_human not in recent:
            messages_to_send = [system_msg, first_human] + recent
        else:
            messages_to_send = [system_msg] + recent
        logger.info(
            "orchestrator_context_trimmed",
            original=len(all_messages),
            trimmed=len(messages_to_send) - 1,
            tenant=tenant_id,
        )
    else:
        messages_to_send = [system_msg] + all_messages

    # ── ROTACIÓN AUTOMÁTICA EN 429 ────────────────────────────────
    response = None
    for _attempt in range(_MAX_MODEL_ROTATIONS):
        llm = await get_orchestrator_llm()
        llm_with_tools = llm.bind_tools(ORCHESTRATOR_TOOLS)
        model_id = getattr(llm, "_laika_model_id", "unknown")
        try:
            response = await llm_with_tools.ainvoke(messages_to_send, config=config)

            # Conteo real de tokens (LiteLLM / OpenAI-compatible response)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                total_tokens = (
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                )
                await record_model_usage(model_id, total_tokens)
            else:
                await record_model_usage(model_id, 0)
            break
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                logger.warning("orchestrator_429_cooldown", model=model_id, attempt=_attempt + 1)
                await set_model_cooldown(model_id, seconds=60)
                if _attempt < _MAX_MODEL_ROTATIONS - 1:
                    continue
            raise

    if response is None:
        raise RuntimeError("Todos los modelos de orchestration agotados")

    logger.info(
        "orchestrator_node_completed",
        tenant=tenant_id,
        has_tool_calls=bool(getattr(response, "tool_calls", None)),
    )

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

