from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime
from structlog import get_logger

from src.api.schemas.requests import N8NWebhookPayload, N8NAcceptanceResponse
from src.worker.tasks import process_agentic_workflow_celery
from src.core.security import verify_token
from src.core.tenant_ratelimit import check_tenant_rate_limit

logger = get_logger("laika_webhook")
router = APIRouter(prefix="/v1/hooks", tags=["N8N Ingestion"])

# ==========================================
# ROUTER: EL PORTAL DE ENTRADA (CONTROL INVERSO)
# ==========================================


@router.post("/n8n", response_model=N8NAcceptanceResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_n8n_webhook(
    payload: N8NWebhookPayload,
    claims: dict = Depends(verify_token),
):
    """
    Gateway de recepcion Omnicanal B2B.

    1. Verifica JWT Bearer token (verify_token dependency).
    2. Valida cross-tenant: el tenant_id del JWT debe coincidir con el del payload.
    3. Aplica rate limiting por tenant (sliding window Redis, 429 + Retry-After).
    4. Delega la consulta pesada a Celery via BackgroundTask.
    5. Devuelve 202 inmediato para no causar Timeout en N8N (Fire and forget).

    Token de desarrollo: `python -c "from src.core.security import generate_dev_token; print(generate_dev_token('mi_tenant'))"` 
    """

    # --- GUARDA CROSS-TENANT ---
    # Si el JWT incluye tenant_id (tokens de producción), lo comparamos con el payload.
    # Tokens de desarrollo (sub="dev") sin tenant_id en el claim pasan sin restricción.
    jwt_tenant = claims.get("tenant_id")
    if jwt_tenant and jwt_tenant != payload.tenant_id:
        logger.warning(
            "cross_tenant_attempt",
            jwt_tenant=jwt_tenant,
            payload_tenant=payload.tenant_id,
            jwt_sub=claims.get("sub"),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El tenant_id del token no coincide con el del payload.",
        )

    # --- RATE LIMITING POR TENANT ---
    await check_tenant_rate_limit(payload.tenant_id)

    logger.info(
        "incoming_webhook",
        tenant_id=payload.tenant_id,
        thread=payload.thread_id,
        jwt_sub=claims.get("sub"),
    )

    # Despachamos al motor cognitivo en la cola Celery y capturamos el task_id.
    task = process_agentic_workflow_celery.apply_async(
        kwargs={"payload_dict": payload.model_dump()}
    )
    logger.info("task_dispatched", task_id=task.id, tenant_id=payload.tenant_id)

    return N8NAcceptanceResponse(
        status="Accepted",
        message="Laika se encuentra evaluando la operacion asincronamente.",
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id,
        task_id=task.id,
    )

