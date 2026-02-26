from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from src.core.state import LaikaState
from src.core.config import settings
from src.brain.workflows.router import router_node
from src.brain.workflows.orchestrator import orchestrator_node
from src.brain.workflows.evaluator import evaluator_node
from src.brain.workflows.casual import casual_node
from src.brain.llm_proxy import get_langfuse_callback
from structlog import get_logger

logger = get_logger("laika_graph")

# ==========================================
# EL CEREBRO ENSAMBLADO (LANGGRAPH)
# ==========================================

def should_route_complex(state: LaikaState) -> str:
    """Decisión condicional post-Router."""
    if state.get("current_intent") == "casual":
        return "casual_end"
    return "complex_orchestration"

def compile_laika_graph():
    """Compila visualmente los nodos agénticos en un StateGraph."""
    logger.info("compiling_static_langgraph")
    
    builder = StateGraph(LaikaState)
    
    builder.add_node("router", router_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("casual", casual_node)
    
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        should_route_complex,
        {
            "casual_end": "casual",
            "complex_orchestration": "orchestrator"
        }
    )
    builder.add_edge("orchestrator", "evaluator")
    builder.add_edge("evaluator", END)
    builder.add_edge("casual", END)
    
    return builder

# El builder se compila una sola vez al cargar el módulo (reutilizado por todos los workers)
laika_graph_builder = compile_laika_graph()


# ==========================================
# EJECUCIÓN CON PERSISTENCIA POSTGRES
# ==========================================
async def invoke_agent(
    tenant_id: str,
    thread_id: str,
    payload_msg: str,
    channel: str = "unknown",
    extra_metadata: dict | None = None,
):
    """
    Punto de entrada para Celery.
    Gestiona el ciclo de vida completo de una solicitud:
      1. Crea el CallbackHandler de Langfuse con contexto por-request
      2. Conecta con Postgres vía psycopg async pool para el checkpointer
      3. Ejecuta el grafo LangGraph con callbacks inyectados
      4. Despacha la respuesta al webhook de n8n
      5. Hace flush explícito de Langfuse para garantizar envío de spans

    El @observe decorator de Langfuse en este punto no es necesario porque
    el CallbackHandler creado aquí ya gestiona el trace raíz del SDK de LangChain.
    El flush explícito al final garantiza que todos los spans del trace sean
    enviados aunque el worker de Celery termine antes del flush automático.
    """
    # 1. Handler de Langfuse por request — contexto tenant/thread/canal
    langfuse_handler = get_langfuse_callback(
        tenant_id,
        thread_id,
        channel=channel,
        extra_metadata=extra_metadata,
    )
    callbacks = [langfuse_handler] if langfuse_handler else []

    # 2. Config de LangGraph con callbacks + metadata de Langfuse v3
    config = {
        "configurable": {
            "thread_id": thread_id,
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

    # 3. Pool de conexiones: autocommit=True previene locks en tablas de checkpoint
    async with AsyncConnectionPool(
        conninfo=settings.psycopg_database_url,
        max_size=10,
        kwargs={"autocommit": True},
    ) as pool:

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        agent_runtime = laika_graph_builder.compile(checkpointer=checkpointer)

        from langchain_core.messages import HumanMessage
        input_state = {"messages": [HumanMessage(content=payload_msg)]}

        try:
            # 4. Ejecución del grafo — Langfuse traza cada nodo como un span
            final_state = await agent_runtime.ainvoke(input_state, config)

            last_msg = final_state["messages"][-1]
            response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

            logger.info("laika_finished_thinking", tenant=tenant_id, thread=thread_id)

        except Exception as e:
            # Registrar el error como score visible en Langfuse dashboard.
            # Así los fallos son consultables por trace_id en vez de solo en Docker logs.
            if langfuse_handler is not None:
                from src.brain.llm_proxy import register_trace_score
                register_trace_score(
                    "runtime_error",
                    1.0,
                    langfuse_handler,
                    comment=f"{type(e).__name__}: {str(e)}",
                )
            logger.exception("agent_runtime_failed", error=str(e), tenant=tenant_id)
            raise
        finally:
            # 5. Flush explícito de Langfuse.
            # IMPORTANTE: usamos `langfuse_handler.langfuse` (el cliente del handler)
            # en lugar de crear una nueva instancia con `Langfuse()`.
            # Crear una nueva instancia desconectada del trace activo llama flush
            # sobre un cliente que no conoce los spans de ESTE trace.
            if langfuse_handler is not None:
                try:
                    lf_client = getattr(langfuse_handler, "langfuse", None)
                    if lf_client:
                        lf_client.flush()
                except Exception as flush_err:
                    logger.warning("langfuse_flush_failed", error=str(flush_err))

    # 6. Despacho final al webhook de respuesta en n8n
    import httpx
    webhook_reply_url = f"{settings.N8N_WEBHOOK_URL}webhook/laika-telegram-reply"

    headers = {}
    if settings.N8N_API_KEY.get_secret_value():
        headers["X-N8N-API-KEY"] = settings.N8N_API_KEY.get_secret_value()

    async with httpx.AsyncClient() as client:
        await client.post(
            webhook_reply_url,
            json={
                "tenant_id": tenant_id,
                "thread_id": thread_id,
                "response": response_text,
            },
            headers=headers,
            timeout=10.0,
        )
    logger.info("telegram_reply_dispatched", tenant=tenant_id, thread=thread_id)

