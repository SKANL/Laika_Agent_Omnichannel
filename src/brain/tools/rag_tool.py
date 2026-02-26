from langchain_core.tools import tool
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
async def perform_rag_search(query: str, tenant_id: str) -> str:
    """
    Realiza una busqueda semantica en la base de conocimientos documental de la empresa.
    Usala SIEMPRE que no tengas certeza de una respuesta o necesites extraer
    normativas o datos tecnicos de los manuales del cliente.
    Si los resultados no son relevantes, reformula la query internamente y reintenta.

    Args:
        query: La pregunta que el LLM formula para buscar en la base de datos.
        tenant_id: ID Magico de la empresa que pide la accion.
    """
    from src.brain.embeddings import encode_text

    logger.info("agentic_rag_search_triggered", query=query[:60], tenant_id=tenant_id)

    result = await _rag_search_attempt(query, tenant_id, encode_text)
    if result:
        return result

    # Auto-reformulacion: expandir la query con terminos alternativos
    for attempt in range(1, _MAX_REFORMULATION_RETRIES + 1):
        reformulated = _reformulate_query(query, attempt)
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
    """Reformulacion heuristica simple. En Fase 2 se reemplaza con LLM."""
    prefixes = [
        "informacion sobre ",
        "detalles tecnicos documentados de ",
    ]
    return prefixes[(attempt - 1) % len(prefixes)] + original

