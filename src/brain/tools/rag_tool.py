from langchain_core.tools import tool
from sqlalchemy import select
from src.core.db import AsyncSessionLocal, RAGDocument
from structlog import get_logger

logger = get_logger("laika_rag_tool")

# ==========================================
# AGENTIC RAG (BÚSQUEDA DINÁMICA POR TENANT)
# ==========================================
# Con este decorador "@tool", LangGraph le puede decir  
# al Heavy Lifter (Llama 70b): "Tienes esta función disponible para buscar".

@tool
async def perform_rag_search(query: str, tenant_id: str) -> str:
    """
    Realiza una búsqueda semántica en la base de conocimientos documental de la empresa.
    Úsala SIEMPRE que no tengas certeza de una respuesta o necesites extraer
    normativas o datos técnicos de los manuales del cliente.
    
    Args:
        query: La pregunta que el LLM formula para buscar en la base de datos.
        tenant_id: ID Mágico de la empresa que pide la acción.
    """
    logger.info("agentic_rag_search_triggered", query=query[:30], tenant_id=tenant_id)
    
    # 1. ENTORNO REAL: Aquí se convertiría el "query" (texto String) 
    # a Vector Crudo [0.1, 0.4, ..., 0.6] usando sentence-transformers en la GPU.
    # Simulación temporal del embedding:
    simulated_query_embedding = [0.1] * 384
    
    async with AsyncSessionLocal() as session:
        # 🔴 REGLA 1 B2B RLS (Row-Level Security / Context Filter):
        # Jamás hacer vector search sin limitar al Tenant del Dueño. 
        stmt = (
            select(RAGDocument.content)
            .where(RAGDocument.tenant_id == tenant_id)
            .order_by(RAGDocument.embedding.cosine_distance(simulated_query_embedding))
            .limit(3) # Extraemos solo el Top 3 (Chunks más relevantes)
        )
        
        result = await session.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            return "No se encontró información documental relevante en la memoria de la empresa."
            
        # Componemos el Raw Context para inyectarlo en LangGraph
        compiled_context = "\n".join([f"Fragmento {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
        
        logger.info("agentic_rag_success", chunks_found=len(chunks), tenant_id=tenant_id)
        return compiled_context
