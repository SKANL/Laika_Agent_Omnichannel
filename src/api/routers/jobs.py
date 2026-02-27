# ==========================================
# JOBS ROUTER: ESTADO DE TAREAS ASÍNCRONAS
# ==========================================
# GET /v1/jobs/{task_id} → estado actual del job Celery.
# Permite a n8n (o cualquier cliente) consultar si un job
# está pendiente, en proceso, exitoso o fallido.
# El task_id se expone en los logs del 202 Accepted (campo 'task_id').

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, Any
from structlog import get_logger

from src.core.security import verify_token

logger = get_logger("laika_jobs")
router = APIRouter(prefix="/v1/jobs", tags=["Job Status"])


class JobStatusResponse(BaseModel):
    task_id: str
    status: str            # PENDING | STARTED | SUCCESS | FAILURE | RETRY | REVOKED
    result: Optional[Any] = None
    error: Optional[str] = None


@router.get("/{task_id}", response_model=JobStatusResponse)
async def get_job_status(
    task_id: str,
    claims: dict = Depends(verify_token),
):
    """
    Consulta el estado de un job Celery por su task_id.

    Estados posibles:
      - PENDING  → encolado, aún no iniciado
      - STARTED  → en proceso activo
      - SUCCESS  → completado exitosamente
      - FAILURE  → falló (ver campo 'error')
      - RETRY    → reintentando tras error transitorio
      - REVOKED  → cancelado manualmente

    El task_id es devuelto en el header de logs del 202 Accepted
    y puede ser consultado por n8n para seguimiento asíncrono.
    """
    try:
        from src.worker.celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        status_str = result.status
        job_result = None
        job_error = None

        if result.successful():
            job_result = result.result
        elif result.failed():
            job_error = str(result.result)

        logger.info(
            "job_status_queried",
            task_id=task_id,
            status=status_str,
            tenant=claims.get("tenant_id", "unknown"),
        )

        return JobStatusResponse(
            task_id=task_id,
            status=status_str,
            result=job_result,
            error=job_error,
        )

    except Exception as e:
        logger.error("job_status_query_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error consultando estado del job: {str(e)}",
        )
