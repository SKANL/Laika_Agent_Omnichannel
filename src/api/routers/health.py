# ==========================================
# HEALTH ROUTER: MONITOREO EN TIEMPO REAL
# ==========================================
# Endpoints de observabilidad para operadores y alerting externo.
#
# GET /health           → latido rápido (sin I/O, para Docker/K8s probes)
# GET /health/models    → estado live de cada modelo del pool (Redis)
# GET /health/rotation  → configuración actual del sistema de rotación (YAML)

import os
import yaml
from fastapi import APIRouter, status
from structlog import get_logger

logger = get_logger("laika_health")
router = APIRouter(prefix="/health", tags=["Monitoring"])

_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "models_registry.yaml"
)


@router.get("", status_code=status.HTTP_200_OK)
async def health_check():
    """Latido rápido para orquestadores Docker/K8s (sin I/O)."""
    return {"status": "alive", "agent": "Laika V2 B2B"}


@router.get("/models", status_code=status.HTTP_200_OK)
async def models_status():
    """
    Estado en tiempo real de cada modelo del pool de rotación.

    Para cada model_id devuelve:
      - tpm_used / tpm_limit / tpm_pct   → consumo de tokens en el minuto actual
      - rpm_used / rpm_limit / rpm_pct   → requests en el minuto actual
      - in_cooldown                      → True si recibió un 429 recientemente
      - available                        → True si puede atender requests ahora

    Útil para dashboards de monitoreo, alertas de Grafana o depuración manual.
    """
    try:
        from src.brain.rate_limiter import get_models_status
        models = await get_models_status()
        available_count = sum(1 for m in models.values() if m.get("available", False))
        return {
            "models": models,
            "summary": {
                "total": len(models),
                "available": available_count,
                "limited": len(models) - available_count,
            },
        }
    except Exception as e:
        logger.error("health_models_error", error=str(e))
        return {"models": {}, "error": str(e)}


@router.get("/rotation", status_code=status.HTTP_200_OK)
async def rotation_config():
    """
    Configuración actual del sistema de rotación leída directamente del YAML.
    Muestra qué modelos están activos, sus pesos, límites y estrategia de rotación.
    No requiere Redis — sirve el YAML parseado para diagnóstico rápido.
    """
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        settings_cfg = registry.get("settings", {})
        categories_summary = {}

        for cat_name, cat_data in registry.get("categories", {}).items():
            pool = cat_data.get("pool", [])
            active = [
                {
                    "id": m["id"],
                    "provider": m.get("provider"),
                    "model_id": m.get("model_id"),
                    "weight": m.get("weight", 1),
                    "tool_use": m.get("tool_use", False),
                    "limits": m.get("limits", {}),
                    "notes": m.get("notes", ""),
                }
                for m in pool if m.get("active", False)
            ]
            inactive_ids = [m["id"] for m in pool if not m.get("active", False)]

            categories_summary[cat_name] = {
                "description": cat_data.get("description", ""),
                "rotation_enabled": cat_data.get(
                    "rotation_enabled",
                    settings_cfg.get("rotation_enabled", True),
                ),
                "active_models": active,
                "inactive_model_ids": inactive_ids,
            }

        return {
            "settings": settings_cfg,
            "categories": categories_summary,
        }
    except Exception as e:
        logger.error("health_rotation_error", error=str(e))
        return {"error": str(e)}
