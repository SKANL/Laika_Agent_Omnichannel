import asyncio
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from src.core.state import LaikaState
from src.core.config import settings
from src.brain.workflows.router import router_node
from src.brain.workflows.orchestrator import (
    orchestrator_node,
    orchestrator_tool_node,
    should_use_tools,
)
from src.brain.workflows.evaluator import evaluator_node, route_after_evaluator
from src.brain.workflows.casual import casual_node
from src.brain.workflows.moderation import moderation_node
from src.brain.workflows.planner import planner_node
from src.brain.workflows.clarification import clarification_node
from src.brain.workflows.formatter import formatter_node
from src.brain.workflows.task_dispatcher import task_dispatcher_node
from src.brain.llm_proxy import get_langfuse_callback
from structlog import get_logger

logger = get_logger("laika_graph")

# ==========================================
# CHECKPOINTER SETUP — POR PROCESO (flag thread-safe por asyncio)
# ==========================================
# Cada proceso (FastAPI o Celery worker) llama setup() UNA sola vez.
# setup() es idempotente (CREATE TABLE IF NOT EXISTS) pero tiene overhead.
_setup_done = False
_setup_lock: asyncio.Lock | None = None


async def _ensure_setup(pool: AsyncConnectionPool) -> None:
    """Llama checkpointer.setup() una unica vez por proceso via double-checked locking."""
    global _setup_done, _setup_lock
    if _setup_lock is None:
        _setup_lock = asyncio.Lock()
    if _setup_done:
        return
    async with _setup_lock:
        if not _setup_done:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            _setup_done = True
            logger.info("checkpointer_setup_complete")


# ==========================================
# ENSAMBLADO DEL GRAFO (TOPOLOGIA COMPLETA)
# ==========================================
# Flujo:
#   START -> router
#   router -> [casual | orchestrator]
#   orchestrator -> [tool_node (loop) | evaluator]
#   tool_node -> orchestrator
#   evaluator -> [orchestrator (retry) | END]
#   casual -> END

def _route_after_moderation(state: LaikaState) -> str:
    """Post-moderation: si fue bloqueado terminar inmediatamente, sino continuar al router."""
    if state.get("current_intent") == "blocked":
        return "done"  # -> END
    return "router"


def _should_route_intent(state: LaikaState) -> str:
    """Post-router: desviar trivialidades a casual, aclaraciones a clarification,
    tareas largas a task_dispatcher, investigaciones al planner, resto al orquestador."""
    intent = state.get("current_intent", "")
    if intent == "casual":
        return "casual"
    if intent == "ambiguous":
        return "clarification"
    if intent == "tarea_larga":
        return "task_dispatcher"
    if intent == "investigacion_complex":
        return "planner"
    return "orchestrator"


def compile_laika_graph():
    """Compila el StateGraph de Laika con todos los nodos y aristas."""
    logger.info("compiling_laika_graph")

    builder = StateGraph(LaikaState)

    # --- Nodos ---
    builder.add_node("moderation", moderation_node)
    builder.add_node("router", router_node)
    builder.add_node("clarification", clarification_node)   # interrupt() para peticiones ambiguas
    builder.add_node("planner", planner_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("tool_node", orchestrator_tool_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("casual", casual_node)
    builder.add_node("task_dispatcher", task_dispatcher_node)  # tareas largas
    builder.add_node("formatter", formatter_node)               # formato por canal

    # --- Aristas ---
    # Nueva topología: START → moderation → (block check) → router → intent routing
    builder.add_edge(START, "moderation")

    # Post-moderation: bloqueado → formatter (respuesta de bloqueo) → END, else → router
    builder.add_conditional_edges(
        "moderation",
        _route_after_moderation,
        {"router": "router", "done": END},
    )

    # Post-router: intent routing multi-camino
    builder.add_conditional_edges(
        "router",
        _should_route_intent,
        {
            "casual": "casual",
            "clarification": "clarification",
            "planner": "planner",
            "orchestrator": "orchestrator",
            "task_dispatcher": "task_dispatcher",
        },
    )

    # Clarification: tras aclaración del usuario → volver al router con nueva info
    builder.add_edge("clarification", "router")

    # Plan & Execute: planner siempre entrega su plan al orquestador
    builder.add_edge("planner", "orchestrator")

    # Post-orchestrator: si hay tool_calls → ejecutar herramientas, sino → evaluar
    builder.add_conditional_edges(
        "orchestrator",
        should_use_tools,
        {"use_tools": "tool_node", "evaluate": "evaluator"},
    )

    # Loop de herramientas: tool_node siempre vuelve al orquestador con los resultados
    builder.add_edge("tool_node", "orchestrator")

    # Post-evaluador: si rechazo con reintentos → orquestador, aprobado → formatter
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {"retry_orchestrator": "orchestrator", "done": "formatter"},
    )

    # Todos los caminos terminan en formatter antes de END
    builder.add_edge("casual", "formatter")
    builder.add_edge("task_dispatcher", "formatter")
    builder.add_edge("formatter", END)

    return builder


# Builder estatico (no compilado): reusado por todos los workers del proceso
laika_graph_builder = compile_laika_graph()


# ==========================================
# EJECUCION CON PERSISTENCIA POSTGRES
# ==========================================
async def invoke_agent(
    tenant_id: str,
    thread_id: str,
    payload_msg: str,
    channel: str = "unknown",
    extra_metadata: dict | None = None,
) -> None:
    """
    Punto de entrada para Celery workers y FastAPI backgrounds tasks.

    Ciclo completo:
      0. Semantic cache check — si acierta, responde directo sin tocar el grafo.
      1. Carga TenantConfig desde Postgres e inyecta en RunnableConfig.
      2. Crea CallbackHandler de Langfuse con contexto por-request.
      3. Abre pool Postgres + AsyncPostgresSaver (setup() solo en primera llamada).
      4. Detecta si hay un interrupt() pendiente → resume con Command(resume=...).
         Si no hay interrupt → invoca con el input_state completo.
      5. Extrae formatted_response de state o último AIMessage como fallback.
      6. Almacena la respuesta en semantic cache para futuros aciertos.
      7. Despacha la respuesta al webhook de reply en n8n.
      8. Flush explicito de Langfuse para garantizar envio de spans en workers cortos.
    """
    # ------------------------------------------------------------------
    # 0. SEMANTIC CACHE — cortocircuito previo al grafo (0 costo LLM)
    # ------------------------------------------------------------------
    try:
        from src.brain.embeddings import encode_text
        from src.brain.tools.cache import check_semantic_cache, store_in_semantic_cache

        query_embedding = encode_text(payload_msg)
        cached = await check_semantic_cache(payload_msg, tenant_id, query_embedding)

        if cached:
            logger.info("semantic_cache_shortcut", tenant=tenant_id, thread=thread_id)
            await _dispatch_reply(tenant_id, thread_id, cached)
            return
    except Exception as e:
        logger.warning("semantic_cache_check_failed", error=str(e), tenant=tenant_id)
        query_embedding = None

    # ------------------------------------------------------------------
    # 1. TENANT CONFIG — carga feature flags, backstory y channel_config
    # ------------------------------------------------------------------
    from src.core.db import AsyncSessionLocal
    from src.core.tenant_config import load_tenant_config

    tenant_cfg_dict: dict = {}
    try:
        async with AsyncSessionLocal() as db:
            tc = await load_tenant_config(tenant_id, db)
            if tc:
                if tc.active_intents:
                    tenant_cfg_dict["active_intents"] = tc.active_intents
                if tc.active_tools:
                    tenant_cfg_dict["active_tools"] = tc.active_tools
                if tc.backstory_override:
                    tenant_cfg_dict["backstory_override"] = tc.backstory_override
                if tc.channel_config:
                    tenant_cfg_dict["channel_config"] = tc.channel_config
                logger.info(
                    "tenant_config_loaded",
                    tenant=tenant_id,
                    has_intents=bool(tc.active_intents),
                    has_backstory=bool(tc.backstory_override),
                )
    except Exception as e:
        # Non-blocking: si falla la carga de config, continuar con defaults
        logger.warning("tenant_config_load_failed", error=str(e), tenant=tenant_id)

    # ------------------------------------------------------------------
    # 2. Langfuse CallbackHandler por request
    # ------------------------------------------------------------------
    langfuse_handler = get_langfuse_callback(
        tenant_id, thread_id, channel=channel, extra_metadata=extra_metadata
    )
    callbacks = [langfuse_handler] if langfuse_handler else []

    # ------------------------------------------------------------------
    # 3. Config de LangGraph: thread_id para Checkpointer + tenant_id en configurable
    # ------------------------------------------------------------------
    config = {
        "configurable": {
            "thread_id": f"{tenant_id}::{thread_id}",
            "tenant_id": tenant_id,
            "channel": channel,
            # Datos del tenant inyectados aquí (nunca en el estado visible al LLM)
            **tenant_cfg_dict,
        },
        "callbacks": callbacks,
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": tenant_id,
            "langfuse_tags": ["laika", f"tenant:{tenant_id}", f"channel:{channel}"],
            "langfuse_trace_name": "laika_conversation",
        },
    }

    logger.info("waking_up_laika", tenant=tenant_id, thread=thread_id)

    # ------------------------------------------------------------------
    # 4. Pool Postgres + AsyncPostgresSaver (setup() idempotente 1 vez)
    # ------------------------------------------------------------------
    async with AsyncConnectionPool(
        conninfo=settings.psycopg_database_url,
        max_size=10,
        kwargs={"autocommit": True},
    ) as pool:

        await _ensure_setup(pool)

        checkpointer = AsyncPostgresSaver(pool)
        agent_runtime = laika_graph_builder.compile(checkpointer=checkpointer)

        from langchain_core.messages import HumanMessage

        # ------------------------------------------------------------------
        # DETECCION DE INTERRUPT PENDIENTE
        # Si el checkpoint tiene un interrupt() activo (clarification_node pausó),
        # reanudamos con Command(resume=payload_msg) en lugar del input_state completo.
        # ------------------------------------------------------------------
        graph_input = None
        try:
            current_snapshot = await agent_runtime.aget_state(config)
            pending_interrupts = current_snapshot.tasks
            has_interrupt = any(
                getattr(t, "interrupts", None) for t in pending_interrupts
            )
            if has_interrupt:
                logger.info(
                    "interrupt_detected_resuming",
                    tenant=tenant_id,
                    thread=thread_id,
                )
                graph_input = Command(resume=payload_msg)
        except Exception as snap_err:
            logger.warning("snapshot_check_failed", error=str(snap_err))

        # Si no hay interrupt pendiente → invocar con input_state fresco
        if graph_input is None:
            graph_input = {
                "messages": [HumanMessage(content=payload_msg)],
                "current_intent": "",
                "extracted_entities": {},
                "worker_errors": [],
                "retry_count": 0,
                "last_eval_approved": True,
                "plan": [],
                "clarification_needed": False,
                "background_task_id": None,
                "channel": channel,
                "formatted_response": None,
            }

        response_text = "No se pudo generar una respuesta."

        try:
            final_state = await agent_runtime.ainvoke(graph_input, config)

            # ------------------------------------------------------------------
            # EXTRACCION DE RESPUESTA:
            # 1. Preferir formatted_response (procesado por formatter_node)
            # 2. Fallback al último AIMessage raw
            # 3. Si hay interrupt activo (clarification) → la pregunta ya fue
            #    despachada dentro del grafo; no re-despachar aquí.
            # ------------------------------------------------------------------
            if final_state.get("formatted_response"):
                response_text = final_state["formatted_response"]
            elif final_state.get("__interrupt__"):
                # El grafo pausó en clarification_node — la pregunta fue inyectada
                # como AIMessage en el state. Extraerla para despacharla al usuario.
                interrupt_payload = final_state["__interrupt__"][0].value
                response_text = interrupt_payload.get(
                    "question", "Necesito más información para continuar."
                )
                logger.info(
                    "clarification_question_dispatching",
                    tenant=tenant_id,
                    question=response_text[:80],
                )
            else:
                last_msg = final_state["messages"][-1]
                response_text = (
                    last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                )

            logger.info("laika_finished_thinking", tenant=tenant_id, thread=thread_id)

        except Exception as e:
            if langfuse_handler is not None:
                from src.brain.llm_proxy import register_trace_score
                register_trace_score("runtime_error", 1.0, langfuse_handler,
                                     comment=f"{type(e).__name__}: {str(e)}")
            logger.exception("agent_runtime_failed", error=str(e), tenant=tenant_id)
            raise

        finally:
            if langfuse_handler is not None:
                try:
                    lf_client = getattr(langfuse_handler, "langfuse", None)
                    if lf_client:
                        lf_client.flush()
                except Exception as flush_err:
                    logger.warning("langfuse_flush_failed", error=str(flush_err))

    # ------------------------------------------------------------------
    # 5. Almacenar en semantic cache para futuros aciertos
    # ------------------------------------------------------------------
    try:
        if query_embedding is not None:
            await store_in_semantic_cache(payload_msg, tenant_id, query_embedding, response_text)
    except Exception as e:
        logger.warning("semantic_cache_store_failed", error=str(e), tenant=tenant_id)

    # ------------------------------------------------------------------
    # 6. Despachar respuesta al webhook de n8n
    # ------------------------------------------------------------------
    await _dispatch_reply(tenant_id, thread_id, response_text)



async def _dispatch_reply(tenant_id: str, thread_id: str, response_text: str) -> None:
    """Envia la respuesta final al webhook de reply en n8n."""
    import httpx
    from src.brain.tools.n8n_tool import trigger_dlq_webhook

    # URL configurable via settings (puede incluir sufijo de canal)
    webhook_reply_url = f"{settings.N8N_WEBHOOK_URL}{settings.N8N_REPLY_WEBHOOK_PATH}"

    headers = {}
    if settings.N8N_API_KEY.get_secret_value():
        headers["X-N8N-API-KEY"] = settings.N8N_API_KEY.get_secret_value()

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                webhook_reply_url,
                json={"tenant_id": tenant_id, "thread_id": thread_id, "response": response_text},
                headers=headers,
                timeout=10.0,
            )
        logger.info("reply_dispatched", tenant=tenant_id, thread=thread_id)
    except Exception as e:
        logger.error("dispatch_reply_failed", error=str(e), tenant=tenant_id)
        # DLQ: notificar a n8n para que el usuario nunca quede en visto
        try:
            await trigger_dlq_webhook(
                tenant_id=tenant_id,
                thread_id=thread_id,
                error_msg=f"dispatch_reply_failed: {str(e)}",
            )
        except Exception as dlq_err:
            logger.critical("dlq_also_failed", error=str(dlq_err), tenant=tenant_id)

