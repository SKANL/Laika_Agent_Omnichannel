# ==========================================
# RATE LIMITER: SLIDING WINDOW POR MODELO (REDIS)
# ==========================================
# Tracking de TPM y RPM por model_id (no por proveedor).
# Los límites reales se cargan desde models_registry.yaml al inicio del módulo.
#
# Flujo de protección:
#   1. Antes de invocar el LLM → check_model_available(model_id)
#   2. Después de invocar      → record_model_usage(model_id, tokens)
#   3. Si se recibe 429        → set_model_cooldown(model_id) — excluye el modelo 60s
#
# Redis Keys usadas:
#   rl:{model_id}:tpm:{window}   → tokens en la ventana de 1 min
#   rl:{model_id}:rpm:{window}   → requests en la ventana de 1 min
#   rl:{model_id}:cooldown       → existe si el modelo está en cooldown (TTL)
#   rot:{category}:index         → contador round_robin por categoría

import os
import time
import yaml
from typing import Optional
import redis.asyncio as aioredis
from structlog import get_logger

logger = get_logger("laika_rate_limiter")

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "models_registry.yaml")


def _load_model_limits() -> dict:
    """
    Construye un dict plano {model_id: {tpm, rpm, rpd, tpd, warn_threshold, block_threshold}}
    leyendo todos los pools del YAML. Usado para lookup O(1) en el hot path.
    """
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        logger.error("rate_limiter_registry_load_failed", error=str(e))
        return {}

    settings = registry.get("settings", {})
    global_block = settings.get("block_threshold", 0.90)
    global_warn = settings.get("warn_threshold", 0.85)

    limits: dict = {}
    for cat_data in registry.get("categories", {}).values():
        for model in cat_data.get("pool", []):
            model_id = model.get("id")
            if model_id:
                raw = model.get("limits", {})
                limits[model_id] = {
                    "tpm": raw.get("tpm") or 999999,
                    "rpm": raw.get("rpm") or 999999,
                    "rpd": raw.get("rpd") or 999999,
                    "tpd": raw.get("tpd") or 999999,
                    "block_threshold": global_block,
                    "warn_threshold": global_warn,
                }
    return limits


# Cargado en startup del módulo — sin I/O en el hot path
_MODEL_LIMITS: dict = _load_model_limits()

_redis_client: Optional[aioredis.Redis] = None
# id() del event loop en el que se créo _redis_client.
# Cada asyncio.run() en Celery workers crea un loop nuevo; si cambia,
# el cliente acumula sockets/handles del loop cerrado y lanza
# "Event loop is closed" al intentar usarlos.
_redis_loop_id: Optional[int] = None


def get_redis_client() -> aioredis.Redis:
    """Singleton del cliente Redis asincrono, loop-aware.

    Celery workers ejecutan cada tarea via asyncio.run(), que crea y destruye
    un event loop por invocacion. Si reutilizamos el cliente ligado al loop
    anterior (ya cerrado), obtenemos 'Event loop is closed' en cualquier await.

    Solucion: rastrear el id() del loop con el que se creo el cliente y
    reconstruirlo cuando detectamos que el loop activo es distinto.
    aioredis.from_url() es lazy (no abre sockets hasta el primer await),
    por lo que la reconstruccion es O(1) sin I/O bloqueante.
    """
    global _redis_client, _redis_loop_id

    try:
        current_loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        # Llamado fuera de un contexto async (ej. en startup sincrono).
        # No hay loop activo; si el cliente existe, lo devolvemos tal cual.
        # Se reconstruira en la primera llamada dentro de un asyncio.run().
        current_loop_id = None

    if _redis_client is None or (current_loop_id is not None and _redis_loop_id != current_loop_id):
        from src.core.config import settings
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        _redis_loop_id = current_loop_id
        logger.debug("redis_client_recreated", loop_id=current_loop_id)

    return _redis_client


# ─────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────

async def record_model_usage(model_id: str, token_count: int = 0) -> dict:
    """
    Registra el consumo de un modelo en la ventana de 1 minuto.
    Llamar DESPUÉS de una invocación al LLM.
    Devuelve métricas de consumo actuales para logging.
    """
    client = get_redis_client()
    window = int(time.time() // 60)

    tpm_key = f"rl:{model_id}:tpm:{window}"
    rpm_key = f"rl:{model_id}:rpm:{window}"

    try:
        pipe = client.pipeline()
        pipe.incrby(tpm_key, max(token_count, 0))
        pipe.incr(rpm_key)
        pipe.expire(tpm_key, 120)  # TTL 2min para manejar skew de reloj
        pipe.expire(rpm_key, 120)
        results = await pipe.execute()

        tpm_used = results[0]
        rpm_used = results[1]

        lim = _MODEL_LIMITS.get(model_id, {})
        tpm_limit = lim.get("tpm", 999999)
        rpm_limit = lim.get("rpm", 999999)
        warn_t = lim.get("warn_threshold", 0.85)

        tpm_pct = tpm_used / tpm_limit
        rpm_pct = rpm_used / rpm_limit

        if tpm_pct > warn_t or rpm_pct > warn_t:
            logger.warning(
                "model_rate_limit_warning",
                model_id=model_id,
                tpm_pct=round(tpm_pct * 100, 1),
                rpm_pct=round(rpm_pct * 100, 1),
            )

        return {
            "model_id": model_id,
            "tpm_used": tpm_used, "tpm_limit": tpm_limit,
            "rpm_used": rpm_used, "rpm_limit": rpm_limit,
            "tpm_pct": round(tpm_pct * 100, 1),
            "rpm_pct": round(rpm_pct * 100, 1),
        }

    except Exception as e:
        logger.error("rate_limiter_redis_error", model_id=model_id, error=str(e))
        return {"model_id": model_id, "tpm_used": 0, "rpm_used": 0}


# Alias de compatibilidad con código anterior (por proveedor → ignorado ahora)
async def record_usage(provider: str, token_count: int = 0) -> dict:
    """Legacy alias — preferir record_model_usage(model_id, tokens)."""
    return await record_model_usage(f"legacy_{provider}", token_count)


async def check_model_available(model_id: str) -> bool:
    """
    Retorna True si el modelo puede atender un nuevo request.
    Verifica:
      1. Cooldown activo (set por set_model_cooldown tras un 429)
      2. TPM/RPM por encima del block_threshold en la ventana actual
    """
    client = get_redis_client()
    window = int(time.time() // 60)

    try:
        # 1. Cooldown check (O(1), no pipeline necesario)
        in_cooldown = await client.exists(f"rl:{model_id}:cooldown")
        if in_cooldown:
            logger.debug("model_in_cooldown", model_id=model_id)
            return False

        # 2. Sliding window check
        tpm_used = int(await client.get(f"rl:{model_id}:tpm:{window}") or 0)
        rpm_used = int(await client.get(f"rl:{model_id}:rpm:{window}") or 0)

        lim = _MODEL_LIMITS.get(model_id, {})
        block_t = lim.get("block_threshold", 0.90)
        tpm_limit = lim.get("tpm", 999999)
        rpm_limit = lim.get("rpm", 999999)

        available = (tpm_used / tpm_limit < block_t) and (rpm_used / rpm_limit < block_t)

        if not available:
            logger.warning(
                "model_blocked_by_rate_limit",
                model_id=model_id,
                tpm_pct=round(tpm_used / tpm_limit * 100, 1),
                rpm_pct=round(rpm_used / rpm_limit * 100, 1),
            )

        return available

    except Exception as e:
        logger.error("rate_check_failed", model_id=model_id, error=str(e))
        return True  # Fail-open: si Redis cae, no bloquear el agente


# Legacy alias
async def check_provider_available(provider: str) -> bool:
    """Legacy alias — preferir check_model_available(model_id)."""
    return True


async def set_model_cooldown(model_id: str, seconds: int = 60) -> None:
    """
    Pone el modelo en cooldown durante `seconds` segundos.
    Llamar cuando la API devuelva un 429 real para excluir el modelo
    del pool de rotación sin esperar a que el sliding window lo detecte.
    El TTL de Redis gestiona la expiración automáticamente.
    """
    client = get_redis_client()
    try:
        await client.set(f"rl:{model_id}:cooldown", "1", ex=seconds)
        logger.info("model_cooldown_set", model_id=model_id, seconds=seconds)
    except Exception as e:
        logger.error("set_model_cooldown_failed", model_id=model_id, error=str(e))


async def get_models_status() -> dict:
    """
    Devuelve el estado live de todos los modelos registrados.
    Expuesto en GET /health/models para monitoreo.
    """
    window = int(time.time() // 60)
    client = get_redis_client()
    status = {}

    for model_id, lim in _MODEL_LIMITS.items():
        try:
            tpm_used = int(await client.get(f"rl:{model_id}:tpm:{window}") or 0)
            rpm_used = int(await client.get(f"rl:{model_id}:rpm:{window}") or 0)
            in_cooldown = bool(await client.exists(f"rl:{model_id}:cooldown"))

            tpm_limit = lim.get("tpm", 1)
            rpm_limit = lim.get("rpm", 1)
            block_t = lim.get("block_threshold", 0.90)

            is_blocked = (tpm_used / tpm_limit >= block_t) or (rpm_used / rpm_limit >= block_t)

            status[model_id] = {
                "tpm_used": tpm_used, "tpm_limit": tpm_limit,
                "tpm_pct": round(tpm_used / tpm_limit * 100, 1),
                "rpm_used": rpm_used, "rpm_limit": rpm_limit,
                "rpm_pct": round(rpm_used / rpm_limit * 100, 1),
                "in_cooldown": in_cooldown,
                "available": not in_cooldown and not is_blocked,
            }
        except Exception:
            status[model_id] = {"error": "redis_unavailable"}

    return status


# Legacy alias
async def get_usage_summary() -> dict:
    """Legacy alias → get_models_status()."""
    return await get_models_status()
