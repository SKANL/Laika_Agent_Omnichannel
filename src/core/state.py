from typing import Annotated, Dict, Any, List, Optional
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

    # === Control del Loop Evaluador-Optimizador ===
    # Incrementado por evaluator_node cada vez que rechaza un borrador.
    # Permite al router condicional cortar el ciclo de reintentos (máx. 2).
    retry_count: int

    # Flag que el evaluator_node establece en False cuando rechaza el borrador.
    # El borde condicional post-evaluador lo usa para decidir si volver al orquestador.
    last_eval_approved: bool

    # === Plan de Ejecución (Orchestrator-Worker / Plan & Execute) ===
    # Generado por planner_node para intenciones 'investigacion_complex'.
    # Cada elemento es un sub-objetivo que el orquestador debe resolver con sus herramientas.
    # El orquestador lo lee como hoja de ruta para guiar sus tool_calls.
    plan: List[str]

    # === Nodo de Clarificación (Human-in-the-Loop) ===
    # Establecido en True por router_node cuando la intención es "ambiguous".
    # clarification_node usa interrupt() para pausar el grafo y preguntar al usuario.
    # Se restablece a False una vez que el usuario responde.
    clarification_needed: bool

    # === Tareas en Background ===
    # ID de la tarea Celery para intenciones de larga duración ("tarea_larga").
    # Permite al usuario consultar el estado en /v1/jobs/{task_id}.
    background_task_id: Optional[str]

    # === Canal de Salida ===
    # Canal de comunicación detectado en el payload de entrada.
    # Usado por formatter_node para adaptar el formato de la respuesta.
    # Valores esperados: "telegram", "whatsapp", "slack", "email", "api"
    channel: str

    # === Respuesta Formateada ===
    # Generada por formatter_node a partir del último AIMessage.
    # Usada como respuesta final en lugar del contenido raw del mensaje.
    formatted_response: Optional[str]

    # NOTA: Los `tenant_id` y `thread_id` NO existen dentro del State como variables libres,
    # van inyectados en el `metadata` rígido del `RunnableConfig` que invoca al grafo
    # para asegurar que el LLM nunca pueda adulterarlos, previniendo fuga trans-tenant.
