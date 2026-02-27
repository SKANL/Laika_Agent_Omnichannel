from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, Optional

# ==========================================
# CONTRATO DE API: INTERCEPCIÓN B2B
# ==========================================
# Estos schemas son la única vía de entrada al Gateway Laika.
# Validan obligatoriamente que la red externa (n8n) mande los IDs
# de Tenant para no arriesgarnos a un Data Leak.

class N8NWebhookPayload(BaseModel):
    """
    Molde JSON inquebrantable para recibir webhooks de N8N.
    Si N8N olvida mandar el tenant_id, FastAPI rechazará el POST
    con un 422 Unprocessable Entity instantáneo.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_id": "org_empresaXYZ",
                "thread_id": "wa_5215551234",
                "channel": "whatsapp",
                "user_query": "¿Tienen stock del producto 10?",
                "metadata": {
                    "user_name": "Juan Perez",
                    "n8n_execution_id": "994"
                }
            }
        }
    )

    tenant_id: str = Field(
        ..., 
        description="ID obligatorio del inquilino/empresa para Data Isolation en pgvector.",
        min_length=3
    )
    thread_id: str = Field(
        ..., 
        description="Identificador del chat o sesión para que LangGraph recupere el State."
    )
    channel: str = Field(
        ..., 
        description="Canal de origen normalizado (ej. whatsapp, slack, web)."
    )
    user_query: str = Field(
        ..., 
        description="El mensaje crudo o instrucción del cliente/usuario."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Datos dinámicos asíncronos para enriquecer Prompts."
    )

class N8NAcceptanceResponse(BaseModel):
    """
    Respuesta Fire-and-Forget que FastAPI
    devuelve inmediatamente (HTTP 202) para descolgar a N8N.
    """
    status: str = "Accepted"
    message: str = "Worker has accepted the task."
    thread_id: str
    tenant_id: str
    # ID de la tarea Celery encolada. El cliente puede consultarlo via GET /v1/jobs/{task_id}.
    task_id: Optional[str] = None
