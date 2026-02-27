"""
Tests para src/brain/rate_limiter.py

Cubre: check_model_available, set_model_cooldown, record_model_usage.
Todos los tests mockear Redis para aislamiento total.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_check_model_available_no_cooldown_no_usage():
    """El modelo está disponible cuando no hay cooldown ni consumo registrado."""
    with patch("src.brain.rate_limiter.get_redis_client") as mock_factory:
        client = AsyncMock()
        client.exists = AsyncMock(return_value=0)
        client.get = AsyncMock(return_value=None)
        mock_factory.return_value = client

        from src.brain.rate_limiter import check_model_available
        result = await check_model_available("groq_llama31_8b")

        assert result is True
        client.exists.assert_called_once()


@pytest.mark.asyncio
async def test_check_model_available_in_cooldown():
    """El modelo no está disponible cuando hay cooldown activo."""
    with patch("src.brain.rate_limiter.get_redis_client") as mock_factory:
        client = AsyncMock()
        client.exists = AsyncMock(return_value=1)  # cooldown activo
        mock_factory.return_value = client

        from src.brain.rate_limiter import check_model_available
        result = await check_model_available("groq_llama31_8b")

        assert result is False


@pytest.mark.asyncio
async def test_check_model_available_tpm_at_limit():
    """El modelo no está disponible cuando ha superado el block_threshold de TPM."""
    with patch("src.brain.rate_limiter.get_redis_client") as mock_factory:
        client = AsyncMock()
        client.exists = AsyncMock(return_value=0)
        # groq_llama31_8b tiene tpm=6000; 0.91 * 6000 = 5460 > block_threshold (0.90 → 5400)
        client.get = AsyncMock(side_effect=[str(5460), str(0)])
        mock_factory.return_value = client

        from src.brain.rate_limiter import check_model_available
        result = await check_model_available("groq_llama31_8b")

        assert result is False


@pytest.mark.asyncio
async def test_record_model_usage_returns_metrics_dict():
    """record_model_usage devuelve un dict con tpm_used, rpm_used y porcentajes."""
    with patch("src.brain.rate_limiter.get_redis_client") as mock_factory:
        client = AsyncMock()
        pipe = AsyncMock()
        pipe.__aenter__ = AsyncMock(return_value=pipe)
        pipe.__aexit__ = AsyncMock(return_value=None)
        # incrby → 500 tokens, incr → 5 requests, expire × 2
        pipe.execute = AsyncMock(return_value=[500, 5, True, True])
        pipe.incrby = AsyncMock()
        pipe.incr = AsyncMock()
        pipe.expire = AsyncMock()
        client.pipeline = MagicMock(return_value=pipe)
        mock_factory.return_value = client

        from src.brain.rate_limiter import record_model_usage
        result = await record_model_usage("groq_llama31_8b", 500)

        assert result["tpm_used"] == 500
        assert result["rpm_used"] == 5
        assert "tpm_pct" in result
        assert "rpm_pct" in result


@pytest.mark.asyncio
async def test_set_model_cooldown_sets_redis_key_with_ttl():
    """set_model_cooldown crea la key Redis con el TTL correcto."""
    with patch("src.brain.rate_limiter.get_redis_client") as mock_factory:
        client = AsyncMock()
        client.set = AsyncMock(return_value=True)
        mock_factory.return_value = client

        from src.brain.rate_limiter import set_model_cooldown
        await set_model_cooldown("groq_llama31_8b", seconds=60)

        client.set.assert_called_once_with("rl:groq_llama31_8b:cooldown", "1", ex=60)


@pytest.mark.asyncio
async def test_rate_limiter_redis_failure_returns_true():
    """Ante fallo de Redis, check_model_available retorna True (fail-open)."""
    with patch("src.brain.rate_limiter.get_redis_client") as mock_factory:
        client = AsyncMock()
        client.exists = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_factory.return_value = client

        from src.brain.rate_limiter import check_model_available
        result = await check_model_available("groq_llama31_8b")

        assert result is True  # fail-open
