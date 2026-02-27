# ==========================================
# WEB SEARCH TOOL: BÚSQUEDA EN TIEMPO REAL (TAVILY)
# ==========================================
# Herramienta que el Orquestador puede invocar cuando necesita datos
# que no están en la memoria documental (RAG) del cliente:
# precios de mercado, noticias, normativas públicas, etc.
#
# El @tool decorator lo expone automáticamente a LangGraph para
# que el LLM decida cuándo llamarlo (Agentic RAG complementario).

import os
from langchain_core.tools import tool
from structlog import get_logger

logger = get_logger("laika_web_search")


@tool
async def web_search(query: str) -> str:
    """
    Realiza una búsqueda web en tiempo real para obtener información actualizada
    que no está en la base de conocimientos interna de la empresa.
    Úsala para: precios de mercado, noticias recientes, información pública,
    regulaciones, o cualquier dato que el RAG interno no tiene.

    Args:
        query: La pregunta o término de búsqueda a investigar en internet.
    """
    logger.info("web_search_triggered", query=query[:60])

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        from src.core.config import settings

        api_key = settings.TAVILY_API_KEY.get_secret_value()
        if not api_key:
            return "Error: TAVILY_API_KEY no está configurada. Búsqueda web no disponible."

        # Pasamos la API key directamente al constructor (no via os.environ)
        # para evitar race conditions con workers concurrentes
        search_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            tavily_api_key=api_key,
        )

        results = await search_tool.ainvoke(query)

        if isinstance(results, list):
            formatted = "\n\n".join([
                f"Fuente: {r.get('url', 'N/A')}\n{r.get('content', r.get('snippet', ''))}"
                for r in results[:3]
            ])
        else:
            formatted = str(results)

        logger.info("web_search_success", query=query[:40], results_count=len(results) if isinstance(results, list) else 1)
        return formatted or "Sin resultados relevantes encontrados."

    except ImportError:
        return "Error: langchain_community no instalado. `pip install langchain-community`"
    except Exception as e:
        logger.error("web_search_failed", error=str(e))
        return f"Error al realizar búsqueda web: {str(e)}"
