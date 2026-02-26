import litellm
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM
from src.core.config import settings

# ==========================================
# CEREBRO: LITELLM & LANGFUSE TRACING v3
# ==========================================
# Langfuse v3 usa OpenTelemetry internamente.
# El SDK lee LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY y LANGFUSE_BASE_URL
# del entorno automáticamente — estas variables las inyecta docker-compose.
# NO manipulamos os.environ aquí; eso viola el principio de configuración
# centralizada y puede romper procesos forked como los workers de Celery.

# Resiliencia: configuración global de LiteLLM
litellm.request_timeout = 30
litellm.num_retries = 2


def get_routing_llm() -> BaseChatModel:
    """
    Tier 2 (Velocista): Groq llama-3.1-8b-instant.
    Para tareas de bajo TPM: routing, evaluación ligera y charla casual.
    Fallback automático a Cerebras si Groq devuelve 429.
    """
    return ChatLiteLLM(
        model="groq/llama-3.1-8b-instant",
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        temperature=0.0,
        fallbacks=[{"model": "cerebras/llama3.1-8b"}],
    )


def get_orchestrator_llm() -> BaseChatModel:
    """
    Tier 1 (Heavy Lifter): Groq llama-3.3-70b-versatile.
    Para lectura de contextos densos (RAG, planificación compleja).
    Ventana 60K tokens.
    """
    return ChatLiteLLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        temperature=0.2,
        max_tokens=2048,
    )


def get_langfuse_callback(
    tenant_id: str,
    thread_id: str,
    trace_name: str = "laika_conversation",
    channel: str = "unknown",
    extra_metadata: Optional[dict] = None,
) -> Optional[Any]:
    """
    Crea un CallbackHandler de Langfuse v3 por request.
    Cada request obtiene su propia instancia con el contexto correcto:
      - user_id    → tenant_id   (aislamiento B2B en el dashboard)
      - session_id → thread_id   (hilo de conversación, permite ver historial)
      - tags       → identifica el tenant para filtrar en el dashboard
      - metadata   → canal de origen, ambiente y cualquier dato extra del payload

    Non-blocking: si Langfuse no está disponible retorna None.
    El agente continúa funcionando sin observabilidad.
    """
    try:
        from langfuse.langchain import CallbackHandler
        from langfuse.types import TraceContext

        metadata = {
            "environment": "production",
            "channel": channel,
            **(extra_metadata or {}),
        }
        tc = TraceContext(
            user_id=tenant_id,
            session_id=thread_id,
            tags=["laika", f"tenant:{tenant_id}", f"channel:{channel}"],
            name=trace_name,
            metadata=metadata,
        )
        return CallbackHandler(trace_context=tc)
    except Exception as e:
        import traceback
        import structlog
        structlog.get_logger("laika_llm_proxy").warning(
            "langfuse_callback_handler_init_failed",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        return None


def register_trace_score(
    name: str,
    value: Any,
    config_or_handler: Any,
    comment: str = "",
    data_type: str = "NUMERIC",
) -> None:
    """
    Registra un score en el trace activo de Langfuse. Función compartida para
    todos los nodos del grafo que necesiten registrar métricas de negocio.

    Acepta tanto un RunnableConfig (dentro de un nodo LangGraph) como un
    CallbackHandler directo.

    Casos de uso:
      - Score categórico del intent: register_trace_score("intent", "soporte", config, data_type="CATEGORICAL")
      - Score de error:              register_trace_score("runtime_error", 1.0, handler, comment=str(e))
      - Score de latencia:           register_trace_score("latency_ms", 1540.0, config)

    Non-blocking: nunca rompe el flujo del agente.
    """
    try:
        from langfuse import get_client

        # Acepta RunnableConfig (dict con "callbacks") o un handler directo
        if hasattr(config_or_handler, "last_trace_id"):
            handler = config_or_handler
        else:
            callbacks_raw = (config_or_handler or {}).get("callbacks", [])
            handlers_list = getattr(callbacks_raw, "handlers", None)
            if handlers_list is None:
                try:
                    handlers_list = list(callbacks_raw)
                except TypeError:
                    handlers_list = []
            handler = next(
                (cb for cb in handlers_list if type(cb).__name__ == "LangchainCallbackHandler"),
                None,
            )

        if handler is None:
            return

        trace_id = getattr(handler, "last_trace_id", None)
        if not trace_id:
            return

        get_client().create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type,
        )
    except Exception:
        pass  # Non-blocking siempre

