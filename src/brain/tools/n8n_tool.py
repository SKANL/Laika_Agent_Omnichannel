import httpx
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from src.core.config import settings
from structlog import get_logger

logger = get_logger("laika_n8n_tools")

# ==========================================
# EL BRAZO DE ACCIÓN (CONTROL INVERSO)
# ==========================================

@tool
async def n8n_workflow_execution(workflow_id: str, action_payload: dict, config: RunnableConfig) -> str:
    """
    Pide a n8n que ejecute un Workflow externo (ej. enviar email, crear CRM entry).
    Esta es la única forma en que Laika altera el mundo real.
    
    Args:
        workflow_id: El ID o slug del webhook de n8n a disparar.
        action_payload: Los datos JSON extraídos que N8N necesita para operar.
    """
    # tenant_id se inyecta via RunnableConfig por LangGraph (NO expuesto al LLM)
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")

    logger.info("triggering_n8n_workflow", workflow_id=workflow_id, tenant_id=tenant_id)
    
    # URL Base protegida (apuntando al container de n8n o la nube cliente)
    webhook_url = f"{settings.N8N_WEBHOOK_URL}webhook/{workflow_id}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Preparamos los Headers en caso de que el usuario haya asegurado sus Webhooks n8n
            headers = {}
            if settings.N8N_API_KEY.get_secret_value():
                headers["X-N8N-API-KEY"] = settings.N8N_API_KEY.get_secret_value()

            response = await client.post(
                webhook_url, 
                json={
                    "tenant_id": tenant_id,
                    "cognitive_instruction": action_payload
                },
                headers=headers,
                timeout=15.0 # Cortocircuito si n8n no responde
            )
            response.raise_for_status()
            
            logger.info("n8n_action_success", status=response.status_code, tenant_id=tenant_id)
            return "Workflow ejecutado exitosamente en n8n."
            
    except httpx.HTTPStatusError as e:
        logger.error("n8n_http_error", error=str(e), tenant_id=tenant_id)
        return f"Error HTTP {e.response.status_code} al contactar n8n."
    except Exception as e:
        logger.exception("n8n_connection_failed", error=str(e), tenant_id=tenant_id)
        return f"Fallo catastrófico de conexión con n8n: {str(e)}"

# ------------------------------------------
# FALLA SILENCIOSA (DEAD LETTER QUEUE)
# ------------------------------------------

async def trigger_dlq_webhook(tenant_id: str, thread_id: str, error_msg: str):
    """
    Se invoca de emergencia si un nodo falla críticamente o 
    se agotan los reintentos (Rate Limit 429 persistente).
    
    Notifica a n8n para que N NUNCA deje en visto al usuario.
    """
    logger.critical("triggering_DLQ", tenant_id=tenant_id, thread=thread_id, error=error_msg)
    
    dlq_url = f"{settings.N8N_WEBHOOK_URL}webhook/laika-dlq"
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                dlq_url,
                json={
                    "tenant_id": tenant_id,
                    "thread_id": thread_id,
                    "fatal_error": error_msg,
                    "status": "system_failure"
                },
                headers={"X-N8N-API-KEY": settings.N8N_API_KEY.get_secret_value()} if settings.N8N_API_KEY.get_secret_value() else {},
                timeout=5.0
            )
    except Exception as e:
        # Ya falló el fallback. Dejar registro local ruidoso.
        logger.exception("FATAL_DLQ_DELIVERY_FAILURE", error=str(e))
