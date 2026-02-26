import yaml
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from src.core.state import LaikaState
from src.brain.llm_proxy import get_routing_llm, register_trace_score
from structlog import get_logger

logger = get_logger("laika_router")

# Cargamos the registry en memoria
with open("src/config/prompts_registry.yaml", "r", encoding="utf-8") as file:
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
    
    # Extraemos el último mensaje enviado por el usuario
    # Garantizamos que sea un HumanMessage para evaluarlo
    last_message = state["messages"][-1]
    
    logger.info("router_node_start", tenant=tenant_id, thread=thread_id)

    # 1. Preparamos el Prompt de Routing 
    system_prompt = SystemMessage(content=prompts["system_prompts"]["router_node"])
    
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
        logger.info("router_classification_done", intent=intent, tenant=tenant_id)

        # Registrar el intent como score categórico en Langfuse.
        # Permite filtrar y agrupar traces por tipo de petición en el dashboard.
        register_trace_score("intent", intent, config, data_type="CATEGORICAL")

        return {"current_intent": intent}
        
    except Exception as e:
        logger.error("router_json_parsing_failed", error=str(e), tenant=tenant_id)
        # Fallback de seguridad: Si alucina y rompe el JSON, asumimos charla casual para no romper RAG
        return {"current_intent": "casual"}
