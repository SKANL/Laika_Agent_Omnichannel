"""
clarification.py — Nodo de Clarificación (Human-in-the-Loop)

Implementa el patrón LangGraph interrupt() para pausar el grafo cuando la
intención del usuario es ambigua y necesita una pregunta de aclaración.

Flujo:
  1. router_node clasifica intent = "ambiguous" con una pregunta sugerida.
  2. main_graph enruta a clarification_node.
  3. clarification_node llama interrupt({"question": "..."}).
     → El grafo se PAUSA. invoke_agent recibe result["__interrupt__"] y
       despacha la pregunta al usuario via _dispatch_reply.
  4. El usuario responde → siguiente webhook en el mismo thread_id.
  5. invoke_agent detecta el checkpoint pausado y llama:
       ainvoke(Command(resume=user_response), config)
  6. interrupt() retorna el texto del usuario → clarification_node lo agrega
     como HumanMessage y el router re-clasifica la consulta enriquecida.
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.core.state import LaikaState
from structlog import get_logger

logger = get_logger("laika_clarification")


async def clarification_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Pausa el grafo con una pregunta de aclaración y espera la respuesta del usuario.

    El router_node deposita en extracted_entities["clarification_question"]
    la pregunta sugerida antes de enrutar aquí.
    Si no hay pregunta sugerida, usa un fallback genérico.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")

    # Extraer la pregunta que el router sugirió (o usar fallback)
    question = state.get("extracted_entities", {}).get(
        "clarification_question",
        "¿Podrías darme más detalles sobre lo que necesitas para poder ayudarte mejor?",
    )

    logger.info("clarification_interrupt", tenant=tenant_id, question=question[:80])

    # ============================================================
    # PAUSA DEL GRAFO — el valor de interrupt() se expone en
    # result["__interrupt__"][0].value y es despachado al usuario.
    # El grafo se reanuda con Command(resume=user_response) en el
    # siguiente webhook al mismo thread_id.
    # ============================================================
    user_clarification: str = interrupt({"question": question})

    logger.info(
        "clarification_resumed",
        tenant=tenant_id,
        user_clarification=user_clarification[:80],
    )

    # Agregar la respuesta del usuario como HumanMessage para que el router
    # la use como nueva consulta en el siguiente ciclo del grafo.
    return {
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=user_clarification),
        ],
        "clarification_needed": False,
        "current_intent": "",  # Forzar re-clasificación en el router
    }
