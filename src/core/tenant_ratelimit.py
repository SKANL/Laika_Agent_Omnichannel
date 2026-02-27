# ==========================================
# TENANT RATE LIMITER: CONTROL DE FLOODING
# ==========================================
# Dependency de FastAPI que limita requests por tenant_id usando Redis.
# Protege el webhook de ataques de flooding y abuso de free tier.
#
# Implementación: sliding window de 60 segundos vía Redis INCR.
# Límite configurable via WEBHOOK_RATE_LIMIT_PER_MINUTE en settings.

import time
from fastapi import HTTPException, status
from structlog import get_logger

logger = get_logger("laika_tenant_ratelimit")


async def check_tenant_rate_limit(tenant_id: str) -> None:
    """
    Verifica que el tenant no haya superado el límite de requests por minuto.
    Llamar DESPUÉS de autenticar y extraer el tenant_id del JWT.

    Raises:
        HTTPException 429 si el tenant supera el límite configurado.
    """
    try:
        from src.brain.rate_limiter import get_redis_client
        from src.core.config import settings

        limit = settings.WEBHOOK_RATE_LIMIT_PER_MINUTE
        window = int(time.time() // 60)
        key = f"tenant_rl:{tenant_id}:{window}"

        client = get_redis_client()
        count = await client.incr(key)
        await client.expire(key, 120)  # TTL 2 min para manejar skew de reloj

        if count > limit:
            logger.warning(
                "tenant_rate_limit_exceeded",
                tenant_id=tenant_id,
                count=count,
                limit=limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit excedido. Máximo {limit} requests/min por tenant.",
                headers={"Retry-After": "60"},
            )

    except HTTPException:
        raise
    except Exception as e:
        # Fail-open: si Redis falla, no bloquear el agente
        logger.error("tenant_ratelimit_redis_error", error=str(e), tenant_id=tenant_id)
