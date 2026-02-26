from sqlalchemy import select
from src.core.db import AsyncSessionLocal, SemanticCache
from structlog import get_logger

logger = get_logger("laika_semantic_cache")

# Distancia coseno maxima para considerar un acierto.
# Con vectores normalizados L2: distancia < 0.04 equivale a similitud > 96%.
_CACHE_HIT_THRESHOLD = 0.04

# ==========================================
# CACHE SEMANTICA (EVASION DE LLM PREVENTIVA)
# ==========================================

async def check_semantic_cache(user_query: str, tenant_id: str, query_embedding: list[float]) -> str | None:
    """
    Busca si una pregunta identica en INTENCION VECTORIAL ya fue
    respondida antes para esta MISMA EMPRESA.

    Umbral: similitud coseno > 96% (distancia coseno < 0.04).
    Los embeddings deben estar normalizados L2 (normalize_embeddings=True).

    Ahorro: $0 costo de LLM + latencia 3s -> 50ms por acierto.
    """
    logger.info("checking_semantic_cache", tenant_id=tenant_id)

    async with AsyncSessionLocal() as session:
        # pgvector <-> es distancia coseno.
        # Filtramos PRIMERO por tenant_id (RLS) y luego ordenamos por distancia.
        stmt = (
            select(
                SemanticCache,
                SemanticCache.question_embedding.cosine_distance(query_embedding).label("distance"),
            )
            .where(SemanticCache.tenant_id == tenant_id)
            .order_by(SemanticCache.question_embedding.cosine_distance(query_embedding))
            .limit(1)
        )

        result = await session.execute(stmt)
        row = result.first()

        if row is None:
            logger.info("semantic_cache_miss_empty", tenant_id=tenant_id)
            return None

        closest_match, distance = row
        dist_value = float(distance)

        if dist_value < _CACHE_HIT_THRESHOLD:
            logger.info(
                "semantic_cache_hit",
                tenant_id=tenant_id,
                matched_id=closest_match.id,
                distance=round(dist_value, 4),
                similarity_pct=round((1 - dist_value) * 100, 2),
            )
            return closest_match.pre_computed_answer

        logger.info("semantic_cache_miss", tenant_id=tenant_id,
                    closest_distance=round(dist_value, 4))
        return None


async def store_in_semantic_cache(
    question: str,
    tenant_id: str,
    question_embedding: list[float],
    answer: str,
) -> None:
    """
    Guarda la respuesta generada para reutilizarla en futuros aciertos.
    Llamar DESPUES de una invocacion exitosa al grafo LangGraph.
    """
    async with AsyncSessionLocal() as session:
        try:
            entry = SemanticCache(
                tenant_id=tenant_id,
                question=question,
                question_embedding=question_embedding,
                pre_computed_answer=answer,
            )
            session.add(entry)
            await session.commit()
            logger.info("semantic_cache_stored", tenant_id=tenant_id)
        except Exception as e:
            await session.rollback()
            logger.error("semantic_cache_store_failed", error=str(e), tenant_id=tenant_id)

