from fastapi import APIRouter, BackgroundTasks, HTTPException, status, Depends
from pydantic import ValidationError
from datetime import datetime
from structlog import get_logger

from src.api.schemas.requests import N8NWebhookPayload, N8NAcceptanceResponse
from src.worker.tasks import process_agentic_workflow_celery
from src.core.security import verify_token

logger = get_logger("laika_webhook")
router = APIRouter(prefix="/v1/hooks", tags=["N8N Ingestion"])

# ==========================================
# ROUTER: EL PORTAL DE ENTRADA (CONTROL INVERSO)
# ==========================================

def process_agentic_workflow_in_background(payload: N8NWebhookPayload):
    """
    Despacha el Payload hacia la cola de Redis para que un Worker lo procese.
    """
    logger.info("dispatching_to_celery",
                tenant_id=payload.tenant_id,
                thread=payload.thread_id)

    # Invocamos la tarea de Celery .delay() para enviar al broker Redis
    process_agentic_workflow_celery.delay(payload.model_dump())


@router.post("/n8n", response_model=N8NAcceptanceResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_n8n_webhook(
    payload: N8NWebhookPayload,
    background_tasks: BackgroundTasks,
    claims: dict = Depends(verify_token),
):
    """
    Gateway de recepcion Omnicanal B2B.

    1. Verifica JWT Bearer token (verify_token dependency).
    2. Valida estrictamente que el tenant_id exista en el esquema JSON Pydantic.
    3. Delega la consulta pesada a Celery via BackgroundTask.
    4. Devuelve 202 inmediato para no causar Timeout en N8N (Fire and forget).

    Token de desarrollo: `python -c "from src.core.security import generate_dev_token; print(generate_dev_token('mi_tenant'))"` 
    """
    logger.info("incoming_webhook",
                tenant_id=payload.tenant_id,
                thread=payload.thread_id,
                jwt_sub=claims.get("sub"))

    # Despachamos al motor cognitivo en 2do plano sin bloquear el hilo principal.
    background_tasks.add_task(process_agentic_workflow_in_background, payload)

    # Regresamos el "Ack" instantaneo.
    return N8NAcceptanceResponse(
        status="Accepted",
        message="Laika se encuentra evaluando la operacion asincronamente.",
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id,
    )

