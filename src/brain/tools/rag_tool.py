from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select
from src.core.db import AsyncSessionLocal, RAGDocument
from structlog import get_logger

logger = get_logger("laika_rag_tool")

# Distancia coseno maxima para un chunk util (aprox >= 65% similitud).
# Chunks mas lejanos son ruido - mejor "sin informacion" que alucinar.
_RAG_RELEVANCE_THRESHOLD = 0.35
_MAX_REFORMULATION_RETRIES = 2

# ==========================================
# AGENTIC RAG (BUSQUEDA DINAMICA POR TENANT)
# ==========================================
# Con "@tool", LangGraph expone esta funcion al Heavy Lifter (70B).
# El LLM decide cuando invocarla y formula la query autonomamente.

@tool
async def perform_rag_search(query: str, config: RunnableConfig) -> str:
    """
    Realiza una busqueda semantica en la base de conocimientos documental de la empresa.
    Usala SIEMPRE que no tengas certeza de una respuesta o necesites extraer
    normativas o datos tecnicos de los manuales del cliente.
    Si los resultados no son relevantes, reformula la query internamente y reintenta.

    Args:
        query: La pregunta que el LLM formula para buscar en la base de datos.
    """
    from src.brain.embeddings import encode_text

    # tenant_id se inyecta via RunnableConfig por LangGraph (NO expuesto al LLM)
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "")
    if not tenant_id:
        return "Error de configuracion: tenant_id no disponible en el contexto del agente."

    logger.info("agentic_rag_search_triggered", query=query[:60], tenant_id=tenant_id)

    result = await _rag_search_attempt(query, tenant_id, encode_text)
    if result:
        return result

    # Auto-reformulacion via LLM para expandir la query semanticamente
    for attempt in range(1, _MAX_REFORMULATION_RETRIES + 1):
        reformulated = await _reformulate_with_llm(query)
        logger.info("rag_reformulating", attempt=attempt, new_query=reformulated[:60])
        result = await _rag_search_attempt(reformulated, tenant_id, encode_text)
        if result:
            return f"[Reformulado intento {attempt}]\n{result}"

    logger.info("rag_no_useful_chunks", tenant_id=tenant_id)
    return "No se encontro informacion documental suficientemente relevante en la memoria de la empresa."


async def _rag_search_attempt(query: str, tenant_id: str, encode_fn) -> str | None:
    """Ejecuta una busqueda vectorial y retorna chunks si superan el umbral de relevancia."""
    query_embedding = encode_fn(query)

    async with AsyncSessionLocal() as session:
        # REGLA B2B RLS: SIEMPRE filtrar por tenant_id antes del vector search.
        stmt = (
            select(
                RAGDocument.content,
                RAGDocument.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .where(RAGDocument.tenant_id == tenant_id)
            .order_by(RAGDocument.embedding.cosine_distance(query_embedding))
            .limit(5)
        )

        result = await session.execute(stmt)
        rows = result.all()

    if not rows:
        return None

    # Filtrar chunks que no superen el umbral minimo de relevancia
    relevant_chunks = [
        (content, distance)
        for content, distance in rows
        if float(distance) < _RAG_RELEVANCE_THRESHOLD
    ]

    if not relevant_chunks:
        logger.info("rag_chunks_below_threshold",
                    best_distance=round(float(rows[0][1]), 4) if rows else None)
        return None

    compiled = "\n\n".join([
        f"Fragmento {i+1} (relevancia {round((1 - float(d)) * 100, 1)}%):\n{c}"
        for i, (c, d) in enumerate(relevant_chunks[:3])
    ])

    logger.info("agentic_rag_success",
                chunks_found=len(relevant_chunks), tenant_id=tenant_id,
                top_similarity=round((1 - float(relevant_chunks[0][1])) * 100, 1))
    return compiled


def _reformulate_query(original: str, attempt: int) -> str:
    """Reformulacion heuristica fallback (se usa si _reformulate_with_llm falla)."""
    prefixes = [
        "informacion sobre ",
        "detalles tecnicos documentados de ",
    ]
    return prefixes[(attempt - 1) % len(prefixes)] + original


async def _reformulate_with_llm(original_query: str) -> str:
    """
    Usa el LLM del tier velocista para reformular semanticamente la query RAG.
    Si falla (rate limit, error de conexion), cae al reformulador heuristico.
    """
    try:
        from src.brain.llm_proxy import get_model_for_task
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_model_for_task("routing")  # Tier velocista: rapido y barato
        if llm is None:
            return _reformulate_query(original_query, 1)

        messages = [
            SystemMessage(content=(
                "Eres un experto en reformulacion de queries para busqueda semantica. "
                "Dado el query original, genera UNA version alternativa mas especifica y detallada. "
                "Responde SOLO con el query reformulado, sin explicaciones."
            )),
            HumanMessage(content=f"Query original: {original_query}"),
        ]
        response = await llm.ainvoke(messages)
        reformulated = response.content.strip()
        return reformulated if reformulated else _reformulate_query(original_query, 1)
    except Exception as e:
        logger.warning("rag_llm_reformulation_failed", error=str(e))
        return _reformulate_query(original_query, 1)

