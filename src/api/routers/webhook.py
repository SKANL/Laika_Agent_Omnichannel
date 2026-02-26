from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import ValidationError
from datetime import datetime
from structlog import get_logger

from src.api.schemas.requests import N8NWebhookPayload, N8NAcceptanceResponse

from src.worker.tasks import process_agentic_workflow_celery

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
async def ingest_n8n_webhook(payload: N8NWebhookPayload, background_tasks: BackgroundTasks):
    """
    Gateway de recepción Omnicanal B2B.
    
    1. Valida estrictamente que el tenant_id exista en el esquema JSON Pydantic.
    2. Delega inmediatamente la consulta pesada o la carga del RAG a Background Tasks.
    3. Devuelve 202 Inmediato para no causar Timeout en N8N (Fire and forget).
    """
    
    logger.info("incoming_webhook", tenant_id=payload.tenant_id, thread=payload.thread_id)

    # Despachamos al motor cognitivo en 2do plano sin bloquear el hilo principal.
    background_tasks.add_task(process_agentic_workflow_in_background, payload)

    # Regresamos el "Ack" instantáneo.
    return N8NAcceptanceResponse(
        status="Accepted",
        message="Laika se encuentra evaluando la operación asíncronamente.",
        thread_id=payload.thread_id,
        tenant_id=payload.tenant_id
    )
