import asyncio
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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

def _should_route_intent(state: LaikaState) -> str:
    """Post-router: desviar trivialidades a casual y complejas al orquestador."""
    if state.get("current_intent") == "casual":
        return "casual"
    return "orchestrator"


def compile_laika_graph():
    """Compila el StateGraph de Laika con todos los nodos y aristas."""
    logger.info("compiling_laika_graph")

    builder = StateGraph(LaikaState)

    # --- Nodos ---
    builder.add_node("router", router_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("tool_node", orchestrator_tool_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("casual", casual_node)

    # --- Aristas ---
    builder.add_edge(START, "router")

    # Post-router: intent -> destino
    builder.add_conditional_edges(
        "router",
        _should_route_intent,
        {"casual": "casual", "orchestrator": "orchestrator"},
    )

    # Post-orchestrator: si hay tool_calls -> ejecutar herramientas, sino -> evaluar
    builder.add_conditional_edges(
        "orchestrator",
        should_use_tools,
        {"use_tools": "tool_node", "evaluate": "evaluator"},
    )

    # Loop de herramientas: tool_node siempre vuelve al orquestador con los resultados
    builder.add_edge("tool_node", "orchestrator")

    # Post-evaluador: si rechazo con reintentos -> orquestador, sino -> END
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {"retry_orchestrator": "orchestrator", "done": END},
    )

    builder.add_edge("casual", END)

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
      1. Crea CallbackHandler de Langfuse con contexto por-request.
      2. Abre pool Postgres + AsyncPostgresSaver (setup() solo en primera llamada).
      3. Ejecuta el grafo LangGraph con callbacks inyectados.
      4. Almacena la respuesta en semantic cache para futuros aciertos.
      5. Despacha la respuesta al webhook de reply en n8n.
      6. Flush explicito de Langfuse para garantizar envio de spans en workers cortos.
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
        # Non-blocking: si embeddings/cache falla, continuar con el grafo normal
        logger.warning("semantic_cache_check_failed", error=str(e), tenant=tenant_id)
        query_embedding = None

    # ------------------------------------------------------------------
    # 1. Langfuse CallbackHandler por request
    # ------------------------------------------------------------------
    langfuse_handler = get_langfuse_callback(
        tenant_id, thread_id, channel=channel, extra_metadata=extra_metadata
    )
    callbacks = [langfuse_handler] if langfuse_handler else []

    # ------------------------------------------------------------------
    # 2. Config de LangGraph: thread_id para Checkpointer + tenant_id en configurable
    # ------------------------------------------------------------------
    config = {
        "configurable": {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
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
    # 3. Pool Postgres + AsyncPostgresSaver (setup() idempotente 1 vez)
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

        input_state = {
            "messages": [HumanMessage(content=payload_msg)],
            "current_intent": "",
            "extracted_entities": {},
            "worker_errors": [],
            "retry_count": 0,
            "last_eval_approved": True,
        }

        response_text = "No se pudo generar una respuesta."

        try:
            final_state = await agent_runtime.ainvoke(input_state, config)
            last_msg = final_state["messages"][-1]
            response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            logger.info("laika_finished_thinking", tenant=tenant_id, thread=thread_id)

        except Exception as e:
            if langfuse_handler is not None:
                from src.brain.llm_proxy import register_trace_score
                register_trace_score("runtime_error", 1.0, langfuse_handler,
                                     comment=f"{type(e).__name__}: {str(e)}")
            logger.exception("agent_runtime_failed", error=str(e), tenant=tenant_id)
            raise

        finally:
            # 6. Flush explicito: garantiza envio de spans en workers cortos (Celery)
            if langfuse_handler is not None:
                try:
                    lf_client = getattr(langfuse_handler, "langfuse", None)
                    if lf_client:
                        lf_client.flush()
                except Exception as flush_err:
                    logger.warning("langfuse_flush_failed", error=str(flush_err))

    # ------------------------------------------------------------------
    # 4. Almacenar en semantic cache para futuros aciertos
    # ------------------------------------------------------------------
    try:
        if query_embedding is not None:
            await store_in_semantic_cache(payload_msg, tenant_id, query_embedding, response_text)
    except Exception as e:
        logger.warning("semantic_cache_store_failed", error=str(e), tenant=tenant_id)

    # ------------------------------------------------------------------
    # 5. Despachar respuesta al webhook de n8n
    # ------------------------------------------------------------------
    await _dispatch_reply(tenant_id, thread_id, response_text)


async def _dispatch_reply(tenant_id: str, thread_id: str, response_text: str) -> None:
    """Envia la respuesta final al webhook de reply en n8n."""
    import httpx
    from src.brain.tools.n8n_tool import trigger_dlq_webhook

    webhook_reply_url = f"{settings.N8N_WEBHOOK_URL}webhook/laika-telegram-reply"

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

