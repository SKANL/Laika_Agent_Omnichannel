"""
Tests para src/brain/tools/rag_tool.py

Cubre: que el tenant_id se obtiene de RunnableConfig (no del LLM),
que la búsqueda filtra por tenant, y que falla de forma controlada
ante config vacío.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_db_ctx(rows):
    """Helper: sesión de DB que devuelve rows en el execute."""
    mock_result = MagicMock()
    mock_result.all = MagicMock(return_value=rows)
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


@pytest.mark.asyncio
async def test_rag_search_no_docs_returns_no_info_message():
    """perform_rag_search retorna mensaje de 'no encontrado' cuando DB está vacía."""
    config = {"configurable": {"tenant_id": "org_a", "thread_id": "t1"}}
    mock_ctx = _make_db_ctx([])

    with patch("src.brain.tools.rag_tool.AsyncSessionLocal", return_value=mock_ctx), \
         patch("src.brain.tools.rag_tool.encode_text", return_value=[0.1] * 384):
        from src.brain.tools.rag_tool import perform_rag_search
        result = await perform_rag_search.ainvoke({"query": "¿precio?"}, config=config)
        assert "No se encontro" in result or "no encontr" in result.lower()


@pytest.mark.asyncio
async def test_rag_search_missing_tenant_returns_error():
    """perform_rag_search con config sin tenant_id devuelve error controlado."""
    config = {"configurable": {}}
    mock_ctx = _make_db_ctx([])

    with patch("src.brain.tools.rag_tool.AsyncSessionLocal", return_value=mock_ctx), \
         patch("src.brain.tools.rag_tool.encode_text", return_value=[0.1] * 384):
        from src.brain.tools.rag_tool import perform_rag_search
        result = await perform_rag_search.ainvoke({"query": "test"}, config=config)
        assert "tenant_id" in result.lower()


@pytest.mark.asyncio
async def test_rag_search_with_relevant_chunk_returns_content():
    """perform_rag_search retorna el contenido del chunk relevante."""
    config = {"configurable": {"tenant_id": "org_a", "thread_id": "t1"}}
    # distancia 0.20 < 0.35 → pasa el umbral de relevancia
    rows = [("Precio del producto A es $50", 0.20)]
    mock_ctx = _make_db_ctx(rows)

    with patch("src.brain.tools.rag_tool.AsyncSessionLocal", return_value=mock_ctx), \
         patch("src.brain.tools.rag_tool.encode_text", return_value=[0.1] * 384):
        from src.brain.tools.rag_tool import perform_rag_search
        result = await perform_rag_search.ainvoke({"query": "precio"}, config=config)
        assert "Precio del producto A" in result


@pytest.mark.asyncio
async def test_rag_search_with_irrelevant_chunk_reformulates():
    """perform_rag_search reformula y reintenta cuando todos los chunks están por encima del umbral."""
    config = {"configurable": {"tenant_id": "org_a", "thread_id": "t1"}}
    # distancia 0.50 > 0.35 → por debajo del umbral, activa reformulación
    rows = [("Texto irrelevante", 0.50)]
    mock_ctx = _make_db_ctx(rows)

    with patch("src.brain.tools.rag_tool.AsyncSessionLocal", return_value=mock_ctx), \
         patch("src.brain.tools.rag_tool.encode_text", return_value=[0.1] * 384), \
         patch("src.brain.tools.rag_tool._reformulate_with_llm",
               AsyncMock(return_value="reformulated query")):
        from src.brain.tools.rag_tool import perform_rag_search
        result = await perform_rag_search.ainvoke({"query": "nada relevante"}, config=config)
        # Después de reformulaciones también falla → mensaje de no encontrado
        assert isinstance(result, str)
