from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from src.core.state import LaikaState
from src.brain.llm_proxy import get_routing_llm
from structlog import get_logger
import yaml

logger = get_logger("laika_casual")

# Cargamos the registry en memoria
with open("src/config/prompts_registry.yaml", "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)

async def casual_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patrón: Casual Responder
    Responde a saludos o mensajes irrelevantes de forma cortés,
    usando el LLM veloz (Groq) en lugar del pesado.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    
    logger.info("casual_node_start", tenant=tenant_id)

    # 1. Preparamos el Prompt para la respuesta casual
    system_prompt = SystemMessage(content=prompts["system_prompts"]["casual_node"])
    
    # 2. Invocamos al Tier 2 (Velocista) con rotación automática en 429
    from src.brain.rate_limiter import set_model_cooldown as _cooldown
    response = None
    for _attempt in range(3):
        llm = await get_routing_llm()
        _mid = getattr(llm, "_laika_model_id", "unknown")
        try:
            # CRITIC: pasar `config` propaga los callbacks de Langfuse al LLM.
            response = await llm.ainvoke([system_prompt] + state["messages"], config=config)
            break
        except Exception as _exc:
            _err = str(_exc).lower()
            if "429" in _err or "rate_limit" in _err or "rate limit" in _err:
                logger.warning("casual_429_cooldown", model=_mid, attempt=_attempt + 1)
                await _cooldown(_mid, seconds=60)
                if _attempt < 2:
                    continue
            raise
    if response is None:
        raise RuntimeError("Todos los modelos de routing agotados en casual_node")
    
    logger.info("casual_node_completed", tenant=tenant_id)
    
    # Devolvemos la actualización del State agregando el mensaje generado
    return {"messages": [response]}
