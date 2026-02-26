from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from src.core.state import LaikaState
from src.brain.llm_proxy import get_orchestrator_llm
from structlog import get_logger

logger = get_logger("laika_orchestrator")

async def orchestrator_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patrón 2: Orchestrator-Worker
    Este nodo solo se dispara si el Router clasificó la intención
    como 'cotizacion', 'soporte' o 'investigacion_complex'.
    Se encarga de planear los pasos del RAG o de invocar Tools complejas.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    intent = state.get("current_intent", "unknown")
    
    logger.info("orchestrator_node_start", intent=intent, tenant=tenant_id)
    
    # Instanciamos al "Heavy Lifter" (Cerebras Llama3 70B con 60k TPM)
    # porque este nodo sí va a leer RAG chunks y memoria densa.
    llm = get_orchestrator_llm()
    
    # En un escenario real, aquí inyectaríamos las `Tools` (n8n_tool, rag_tool)
    # usando llm.bind_tools(lista_de_herramientas)
    
    # Por ahora construimos un prompt con contexto dinámico (Prompt Injection)
    context_instruction = f"Atiende a este usuario del Tenant {tenant_id}. Su intención clasificada es: {intent}."
    
    system_msg = SystemMessage(content=context_instruction)
    
    # Le pasamos todo el historial al 70B.
    # CRÍTICO: pasar `config` propaga los callbacks de Langfuse al LLM.
    response = await llm.ainvoke([system_msg] + state["messages"], config=config)
    
    logger.info("orchestrator_node_completed", tenant=tenant_id)
    
    # Todo lo que escupe el Orquestador se anexa a "messages"
    return {"messages": [response]}
