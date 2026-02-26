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
    
    # 2. Invocamos al Tier 2 (Groq/Velocista) pidiendo explícitamente JSON Mode
    llm = get_routing_llm()
    # Forzamos JSON en el Output
    llm_json = llm.bind(response_format={"type": "json_object"})

    # CRÍTICO: pasar `config` propaga los callbacks de Langfuse al LLM.
    # Sin esto, las llamadas al LLM no aparecen como spans en el dashboard.
    response = await llm_json.ainvoke([system_prompt, last_message], config=config)
    
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
