import os
import yaml
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from src.core.state import LaikaState
from src.brain.llm_proxy import get_routing_llm, register_trace_score
from structlog import get_logger

logger = get_logger("laika_router")

# Path absoluto: funciona independientemente del directorio de trabajo (uvicorn o celery)
_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "prompts_registry.yaml")
with open(_PROMPTS_PATH, "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)

async def router_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patrón 1: Routing
    Ahorra el 90% del costo enviando preguntas triviales a un 
    final inmediato, y desviando las complejas al Orquestador pesado.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    thread_id = configurable.get("thread_id", "unknown")
    
    # Si la moderación ya clasificó el intent como 'blocked', no re-clasificamos.
    existing_intent = state.get("current_intent", "")
    if existing_intent == "blocked":
        logger.info("router_skipped_already_blocked", tenant=tenant_id)
        return {"current_intent": "blocked"}

    # Extraemos el último mensaje enviado por el usuario
    last_message = state["messages"][-1]
    
    logger.info("router_node_start", tenant=tenant_id, thread=thread_id)

    # 1. Preparamos el Prompt de Routing — SIN backstory global para evitar
    #    "Context Rot": el router es un clasificador puro, no necesita la persona Laika.
    #    (Karpathy 2025 Context Engineering: inyectar backstory en cada llamada
    #     de routing es ruido que degrada la señal de clasificación y gasta tokens.)
    router_prompt = prompts["system_prompts"]["router_node"]
    system_prompt = SystemMessage(content=router_prompt)
    
    # 2. Invocamos al Tier 2 (Velocista) con rotación automática en 429
    from src.brain.rate_limiter import set_model_cooldown as _cooldown
    response = None
    for _attempt in range(3):
        llm = await get_routing_llm()
        llm_json = llm.bind(response_format={"type": "json_object"})
        _mid = getattr(llm, "_laika_model_id", "unknown")
        try:
            response = await llm_json.ainvoke([system_prompt, last_message], config=config)
            break
        except Exception as _exc:
            _err = str(_exc).lower()
            if "429" in _err or "rate_limit" in _err or "rate limit" in _err:
                logger.warning("router_429_cooldown", model=_mid, attempt=_attempt + 1)
                await _cooldown(_mid, seconds=60)
                if _attempt < 2:
                    continue
            raise
    if response is None:
        raise RuntimeError("Todos los modelos de routing agotados en router_node")
    
    try:
        data = json.loads(response.content)
        intent = data.get("intent", "casual")

        # ================================================================
        # FEATURE FLAGS: Filtrar intents según tenant_config.active_intents
        # Si el tenant tiene una whitelist y el intent no está en ella,
        # redirigir a casual (respuesta amigable en lugar de error).
        # ================================================================
        active_intents = configurable.get("active_intents")  # None = todo activo
        blocked_intents = {"blocked", "casual", "ambiguous"}  # siempre permitidos
        if active_intents and intent not in blocked_intents:
            if intent not in active_intents:
                logger.warning(
                    "intent_blocked_by_feature_flag",
                    intent=intent,
                    tenant=tenant_id,
                    active_intents=active_intents,
                )
                intent = "casual"

        logger.info("router_classification_done", intent=intent, tenant=tenant_id)

        # Registrar el intent como score categórico en Langfuse.
        register_trace_score("intent", intent, config, data_type="CATEGORICAL")

        # Si la petición es ambigua, extraer la pregunta sugerida
        update: dict = {"current_intent": intent}
        if intent == "ambiguous":
            clarification_q = data.get(
                "clarification_question",
                "¿Podrías darme más detalles sobre lo que necesitas?",
            )
            update["extracted_entities"] = {
                **state.get("extracted_entities", {}),
                "clarification_question": clarification_q,
            }
            update["clarification_needed"] = True

        return update

    except Exception as e:
        logger.error("router_json_parsing_failed", error=str(e), tenant=tenant_id)
        # Fallback de seguridad: Si alucina y rompe el JSON, asumimos charla casual
        return {"current_intent": "casual"}
