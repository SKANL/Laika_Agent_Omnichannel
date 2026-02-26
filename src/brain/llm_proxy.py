import litellm
import yaml
import os
import asyncio
from typing import Optional, Any, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM
from src.core.config import settings

# ==========================================
# CEREBRO: LITELLM + ROTACIÓN AUTOMÁTICA FREE TIER
# ==========================================
# La selección de modelo usa un pool por categoría (models_registry.yaml).
# Dos estrategias disponibles:
#   smart        → descarta modelos en cooldown o cerca de sus límites, elige el de
#                  mayor weight disponible (maximiza throughput con los límites reales)
#   round_robin  → turnos equitativos usando un contador en Redis (bueno para RPD)
#
# Los nodos que usan LLM deben llamar a las funciones async:
#   llm = await get_routing_llm()         ← router, evaluador
#   llm = await get_orchestrator_llm()    ← orquestador con bind_tools()
#
# Después de cada invocación real, registrar uso:
#   await record_model_usage(model_id, tokens)
# El model_id usado se puede obtener del atributo _laika_model_id del llm retornado.

litellm.request_timeout = 30
litellm.num_retries = 0  # manejamos reintentos manualmente para aplicar cooldown

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "models_registry.yaml")


def _load_registry() -> dict:
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        import structlog
        structlog.get_logger("laika_llm_proxy").error("models_registry_load_failed", error=str(e))
        return {}


_registry = _load_registry()


# ─────────────────────────────────────────────────────────────────
# Helpers internos de construcción del pool
# ─────────────────────────────────────────────────────────────────

def _build_pool(category: str) -> List[dict]:
    """
    Retorna los modelos activos de una categoría, respetando el rotation_enabled global
    y el override por categoría.
    """
    settings_cfg = _registry.get("settings", {})
    global_rotation = settings_cfg.get("rotation_enabled", True)

    cat = _registry.get("categories", {}).get(category, {})
    cat_rotation = cat.get("rotation_enabled", global_rotation)

    pool = [m for m in cat.get("pool", []) if m.get("active", False)]

    if not cat_rotation or not global_rotation:
        # Rotación desactivada: devolver solo el primero del pool (comportamiento determinista)
        return pool[:1]

    return pool


def _api_key_for(provider: str) -> str:
    if provider == "groq":
        return settings.GROQ_API_KEY.get_secret_value()
    elif provider == "cerebras":
        return settings.CEREBRAS_API_KEY.get_secret_value()
    return ""


def _make_llm(entry: dict, temperature: float = 0.0, max_tokens: Optional[int] = None) -> BaseChatModel:
    """
    Construye un ChatLiteLLM a partir de un entry del pool.
    Adjunta _laika_model_id como atributo para que el nodo pueda registrar uso.
    """
    provider = entry["provider"]
    model_id_str = f"{provider}/{entry['model_id']}"
    api_key = _api_key_for(provider)

    kwargs: dict = {
        "model": model_id_str,
        "api_key": api_key,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    llm = ChatLiteLLM(**kwargs)
    # Adjuntar el id del pool para tracking en rate_limiter
    llm._laika_model_id = entry["id"]   # type: ignore[attr-defined]
    llm._laika_provider = provider       # type: ignore[attr-defined]
    return llm


async def _select_smart(pool: List[dict]) -> dict:
    """
    Estrategia SMART: filtra modelos disponibles (sin cooldown y bajo threshold),
    luego elige el de mayor weight. Si todos están bloqueados, usa el primer activo
    del pool sin filtro (último recurso).
    """
    from src.brain.rate_limiter import check_model_available

    available = []
    for entry in pool:
        if await check_model_available(entry["id"]):
            available.append(entry)

    if not available:
        import structlog
        structlog.get_logger("laika_llm_proxy").warning(
            "all_models_at_capacity_using_fallback",
            pool_ids=[m["id"] for m in pool],
        )
        return pool[0]  # último recurso: usar el primero aunque esté al límite

    # Mayor weight primero
    return max(available, key=lambda m: m.get("weight", 1))


async def _select_round_robin(category: str, pool: List[dict]) -> dict:
    """
    Estrategia ROUND_ROBIN: incrementa un contador en Redis y selecciona por módulo.
    Distribución equitativa de RPD entre los modelos del pool.
    """
    from src.brain.rate_limiter import get_redis_client

    try:
        client = get_redis_client()
        idx = await client.incr(f"rot:{category}:index")
        return pool[(idx - 1) % len(pool)]
    except Exception:
        return pool[0]


async def _get_model_for_category(
    category: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Punto de entrada principal para selección de modelos.
    Construye el pool, aplica la estrategia configurada y devuelve el ChatLiteLLM listo.
    """
    import structlog
    log = structlog.get_logger("laika_llm_proxy")

    pool = _build_pool(category)
    if not pool:
        raise RuntimeError(
            f"No hay modelos activos en la categoría '{category}'. "
            "Verifica models_registry.yaml."
        )

    strategy = _registry.get("settings", {}).get("rotation_strategy", "smart")

    if strategy == "round_robin":
        entry = await _select_round_robin(category, pool)
    else:
        entry = await _select_smart(pool)

    log.info(
        "model_selected",
        category=category,
        strategy=strategy,
        model_id=entry["id"],
        provider=entry["provider"],
    )

    return _make_llm(entry, temperature=temperature, max_tokens=max_tokens)


# ─────────────────────────────────────────────────────────────────
# API PÚBLICA — usada por orchestrator.py, evaluator.py
# ─────────────────────────────────────────────────────────────────

async def get_routing_llm() -> BaseChatModel:
    """
    Tier routing: modelo rápido/ligero para intent classification, small_talk, evaluación.
    Selecciona del pool 'routing' aplicando la estrategia del YAML.
    """
    return await _get_model_for_category("routing", temperature=0.0)


async def get_orchestrator_llm() -> BaseChatModel:
    """
    Tier orchestration: modelo pesado con tool_use para RAG synthesis y razonamiento.
    Selecciona del pool 'orchestration' aplicando la estrategia del YAML.
    """
    return await _get_model_for_category("orchestration", temperature=0.2, max_tokens=2048)


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

