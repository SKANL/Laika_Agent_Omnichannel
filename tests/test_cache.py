"""
Tests para src/brain/tools/cache.py

Cubre: check_semantic_cache (miss vacío, miss por distancia, hit) y store_in_semantic_cache.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_session_ctx(first_return):
    """Helper: construye un AsyncSessionLocal mock que devuelve first_return en .first()."""
    mock_result = MagicMock()
    mock_result.first = MagicMock(return_value=first_return)
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx, mock_session


@pytest.mark.asyncio
async def test_cache_miss_when_no_rows():
    """check_semantic_cache retorna None si la tabla está vacía."""
    mock_ctx, _ = _make_session_ctx(None)

    with patch("src.brain.tools.cache.AsyncSessionLocal", return_value=mock_ctx):
        from src.brain.tools.cache import check_semantic_cache
        result = await check_semantic_cache("¿Cuál es el precio?", "org_x", [0.1] * 384)
        assert result is None


@pytest.mark.asyncio
async def test_cache_miss_when_distance_too_large():
    """check_semantic_cache retorna None si la distancia supera el umbral 0.04."""
    mock_entry = MagicMock()
    mock_entry.pre_computed_answer = "Respuesta vieja"
    mock_entry.id = 1

    mock_ctx, _ = _make_session_ctx((mock_entry, 0.10))  # distancia 0.10 > 0.04

    with patch("src.brain.tools.cache.AsyncSessionLocal", return_value=mock_ctx):
        from src.brain.tools.cache import check_semantic_cache
        result = await check_semantic_cache("¿Cuál es el precio?", "org_x", [0.1] * 384)
        assert result is None


@pytest.mark.asyncio
async def test_cache_hit_returns_precomputed_answer():
    """check_semantic_cache retorna la respuesta precalculada con similitud >96%."""
    mock_entry = MagicMock()
    mock_entry.pre_computed_answer = "El precio es $100"
    mock_entry.id = 42

    mock_ctx, _ = _make_session_ctx((mock_entry, 0.02))  # distancia 0.02 < 0.04 → hit

    with patch("src.brain.tools.cache.AsyncSessionLocal", return_value=mock_ctx):
        from src.brain.tools.cache import check_semantic_cache
        result = await check_semantic_cache("¿Cuál es el precio?", "org_x", [0.1] * 384)
        assert result == "El precio es $100"


@pytest.mark.asyncio
async def test_store_commits_to_db():
    """store_in_semantic_cache hace add + commit en la sesión."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("src.brain.tools.cache.AsyncSessionLocal", return_value=mock_ctx):
        from src.brain.tools.cache import store_in_semantic_cache
        await store_in_semantic_cache("¿Precio?", "org_x", [0.1] * 384, "El precio es $100")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_store_rolls_back_on_exception():
    """store_in_semantic_cache hace rollback si hay error al guardar."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock(side_effect=Exception("DB error"))
    mock_session.rollback = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("src.brain.tools.cache.AsyncSessionLocal", return_value=mock_ctx):
        from src.brain.tools.cache import store_in_semantic_cache
        # No debe lanzar excepción — error silenciado y logueado
        await store_in_semantic_cache("¿Precio?", "org_x", [0.1] * 384, "respuesta")
        mock_session.rollback.assert_called_once()
