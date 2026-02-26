from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ==========================================
# LAIKASTATE: ESTRUCTURA DE MEMORIA GLOBAL
# ==========================================

class LaikaState(TypedDict):
    """
    Representa el estado global (shared state) del grafo conversacional.
    Es la pieza única que viaja de Nodo a Nodo y se respalda en Postgres.
    """
    
    # === Memoria Principal ===
    # add_messages es un reducer que asegura que al hacer state["messages"] = [nuevo_msj],
    # no se sobreescriba, sino que se concatene al final de la lista histórica.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # === Clasificación y Enrutamiento ===
    # Guardado por el Nodo 'Router'. (Ej: 'cotizacion', 'charlas_general', 'queja')
    current_intent: str 
    
    # === Memoria Selectiva Asíncrona (Aislamiento de tareas B2B) ===
    # Cuando un Agente RAG o un Extractor detecta información clave 
    # (ej. "orden_compra": 1234), la deposita aquí, para que n8n pueda leerla 
    # después sin hacer parsing del texto libre de 'messages'.
    extracted_entities: Dict[str, Any]
    
    # === Monitoreo Resiliente (El DLQ Pattern) ===
    # Si N8N rechaza un envío o el API externa truena, los Tools depositarán 
    # la traza de error aquí. Si el listado se evalúa en el "Evaluador" o el worker de Python, 
    # activará la estrategia Dead Letter Queue.
    worker_errors: List[str]
    
    # NOTA: Los `tenant_id` y `thread_id` NO existen dentro del State como variables libres, 
    # van inyectados en el `metadata` rígido del `RunnableConfig` que invoca al grafo
    # para asegurar que el LLM nunca pueda adulterarlos, previniendo fuga trans-tenant.
