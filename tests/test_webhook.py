"""
Tests de integración para el gateway de webhook.

Cubre: 202 con payload válido, 401 sin token, 403 con tenant cruzado,
422 con payload incompleto.

Requiere JWT_SECRET_KEY, GROQ_API_KEY, CEREBRAS_API_KEY en el entorno
(se establecen vía os.environ.setdefault antes del import de app).
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

# Establecer variables mínimas antes de importar la app
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_tests_laika_2026")
os.environ.setdefault("GROQ_API_KEY", "test_groq")
os.environ.setdefault("CEREBRAS_API_KEY", "test_cerebras")


def _make_token(tenant: str = "org_test") -> str:
    """Genera un JWT válido para el tenant dado."""
    from src.core.security import generate_dev_token
    return generate_dev_token(tenant)


@pytest.mark.asyncio
async def test_webhook_returns_202_valid_payload():
    """Gateway retorna 202 con payload válido y JWT correcto del mismo tenant."""
    with patch("src.worker.tasks.process_agentic_workflow_celery.delay", return_value=None):
        from src.main import app
        token = _make_token("org_test")
        payload = {
            "tenant_id": "org_test",
            "thread_id": "test_thread_1",
            "channel": "test",
            "user_query": "Hola, necesito ayuda",
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/v1/hooks/n8n",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "Accepted"
    assert body["tenant_id"] == "org_test"


@pytest.mark.asyncio
async def test_webhook_rejects_missing_token():
    """Gateway retorna 401 cuando no se envía el header Authorization."""
    from src.main import app
    payload = {
        "tenant_id": "org_test",
        "thread_id": "test_thread_1",
        "channel": "test",
        "user_query": "Hola",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/hooks/n8n", json=payload)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_webhook_rejects_cross_tenant_payload():
    """Gateway retorna 403 si el tenant del JWT no coincide con el del payload."""
    from src.main import app
    token = _make_token("org_trustedA")
    payload = {
        "tenant_id": "org_malicious",  # tenant diferente al del JWT
        "thread_id": "test_thread_1",
        "channel": "test",
        "user_query": "Quiero datos de otro tenant",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/hooks/n8n",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_webhook_rejects_missing_tenant_id():
    """Gateway retorna 422 si el payload no incluye tenant_id."""
    from src.main import app
    token = _make_token("org_test")
    payload = {
        # tenant_id ausente
        "thread_id": "test_thread_1",
        "channel": "test",
        "user_query": "Hola",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/hooks/n8n",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 422  # Pydantic validation error
