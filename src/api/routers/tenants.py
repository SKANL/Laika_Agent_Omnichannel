"""
tenants.py — CRUD de TenantConfig

Permite a los administradores del SaaS:
  - Crear un nuevo tenant (POST /v1/tenants)
  - Leer la configuracion de un tenant (GET /v1/tenants/{tenant_id})
  - Actualizar configuracion parcial (PATCH /v1/tenants/{tenant_id})
  - Activar/desactivar tenant (DEL /v1/tenants/{tenant_id})

Todos los endpoints requieren JWT válido.
Un tenant solo puede leer/modificar su propia config (cross-tenant guard).
Tokens con claim `role=admin` pueden gestionar cualquier tenant.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.db import get_db
from src.core.security import verify_token
from src.core.tenant_config import TenantConfig
from structlog import get_logger

logger = get_logger("laika_tenants")
router = APIRouter(prefix="/v1/tenants", tags=["Tenant Config"])


# ==========================================
# SCHEMAS
# ==========================================

class TenantCreateRequest(BaseModel):
    tenant_id: str
    company_name: str
    active_intents: Optional[List[str]] = None
    active_tools: Optional[Dict[str, Any]] = None
    backstory_override: Optional[str] = None
    channel_config: Optional[Dict[str, Any]] = None


class TenantUpdateRequest(BaseModel):
    company_name: Optional[str] = None
    active_intents: Optional[List[str]] = None
    active_tools: Optional[Dict[str, Any]] = None
    backstory_override: Optional[str] = None
    channel_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class TenantResponse(BaseModel):
    tenant_id: str
    company_name: str
    active_intents: Optional[List[str]]
    active_tools: Optional[Dict[str, Any]]
    backstory_override: Optional[str]
    channel_config: Optional[Dict[str, Any]]
    is_active: bool


# ==========================================
# HELPER: Cross-Tenant Guard
# ==========================================

def _assert_can_access(claims: dict, target_tenant_id: str) -> None:
    """Solo admin o el propio tenant puede acceder."""
    role = claims.get("role", "")
    jwt_tenant = claims.get("tenant_id", "")
    if role == "admin":
        return  # admins pueden todo
    if jwt_tenant and jwt_tenant != target_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para acceder a este tenant.",
        )


# ==========================================
# ENDPOINTS
# ==========================================

@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    body: TenantCreateRequest,
    db: AsyncSession = Depends(get_db),
    claims: dict = Depends(verify_token),
):
    """Crea una nueva configuracion de tenant."""
    # Solo admin puede crear tenants nuevos
    if claims.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo administradores pueden crear tenants.",
        )

    # Verificar que no exista ya
    existing = await db.get(TenantConfig, body.tenant_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"El tenant '{body.tenant_id}' ya existe.",
        )

    config = TenantConfig(
        tenant_id=body.tenant_id,
        company_name=body.company_name,
        active_intents=body.active_intents,
        active_tools=body.active_tools,
        backstory_override=body.backstory_override,
        channel_config=body.channel_config,
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)

    logger.info("tenant_created", tenant_id=config.tenant_id, admin=claims.get("sub"))
    return TenantResponse(**config.to_dict())


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    claims: dict = Depends(verify_token),
):
    """Lee la configuracion de un tenant."""
    _assert_can_access(claims, tenant_id)

    config = await db.get(TenantConfig, tenant_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' no encontrado.",
        )
    return TenantResponse(**config.to_dict())


@router.patch("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: str,
    body: TenantUpdateRequest,
    db: AsyncSession = Depends(get_db),
    claims: dict = Depends(verify_token),
):
    """Actualiza parcialmente la configuracion de un tenant."""
    _assert_can_access(claims, tenant_id)

    config = await db.get(TenantConfig, tenant_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' no encontrado.",
        )

    # Solo actualizar los campos que llegaron (PATCH semántico)
    update_data = body.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(config, field, value)

    await db.commit()
    await db.refresh(config)

    logger.info("tenant_updated", tenant_id=tenant_id, fields=list(update_data.keys()))
    return TenantResponse(**config.to_dict())


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    claims: dict = Depends(verify_token),
):
    """
    Desactiva (soft delete) un tenant.
    No borra los datos — solo pone is_active=False para bloquear nuevos mensajes.
    """
    # Solo admin puede desactivar tenants
    if claims.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo administradores pueden desactivar tenants.",
        )

    config = await db.get(TenantConfig, tenant_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' no encontrado.",
        )

    config.is_active = False
    await db.commit()
    logger.info("tenant_deactivated", tenant_id=tenant_id, admin=claims.get("sub"))
