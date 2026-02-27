"""
tenant_config.py — Configuración Plug & Play por Tenant

Modelo SQLAlchemy que define la "personalidad" de cada cliente B2B.
Cada tenant puede sobrescribir:
  - backstory_override : reemplaza global_backstory en todos los nodos
  - active_intents     : whitelist de intents permitidos (feature flags cognitivos)
  - active_tools       : tools opcionales con su config de conexión
  - channel_config     : reglas de formato por canal (Telegram, Slack, Email)
  - company_name       : nombre visible para Laika en sus respuestas

Cargado al inicio de invoke_agent() e inyectado en RunnableConfig.configurable
para que todos los nodos lo lean sin pasarlo como argumento explícito.
"""
from typing import Optional
from sqlalchemy import Column, String, Boolean, JSON, DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db import Base


class TenantConfig(Base):
    __tablename__ = "tenant_configs"

    # PK: mismo valor que tenant_id usado en JWT y RAGDocument
    tenant_id = Column(String(50), primary_key=True)

    # === Identidad ===
    company_name = Column(String(200), nullable=False, default="Empresa")

    # === Feature Flags Cognitivos ===
    # Lista de intents habilitados para este tenant.
    # Si es null o vacío → todos los intents están activos (retrocompatibilidad).
    # Ejemplo: ["cotizacion", "consulta_rag", "casual"]
    active_intents = Column(JSON, nullable=True, default=None)

    # === Tools Opcionales ===
    # Dict de tools activadas con su config de conexión.
    # Ejemplo: {"crm_lookup": {"base_url": "https://crm.acme.com", "api_key": "sk-..."}}
    # Tools universales (rag, web_search, n8n) SIEMPRE activas, no requieren config aquí.
    active_tools = Column(JSON, nullable=True, default=None)

    # === Personalidad del Agente ===
    # Sobrescribe global_backstory si no es null.
    # Permite que cada cliente tenga un "asistente con nombre y personalidad propios".
    backstory_override = Column(String, nullable=True, default=None)

    # === Configuración de Canal ===
    # Reglas de formato por canal.
    # Ejemplo: {"telegram": {"max_chars": 4096, "use_markdown": true},
    #           "email": {"use_html": true, "max_chars": 10000}}
    channel_config = Column(JSON, nullable=True, default=None)

    # === Control ===
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "company_name": self.company_name,
            "active_intents": self.active_intents,
            "active_tools": self.active_tools,
            "backstory_override": self.backstory_override,
            "channel_config": self.channel_config,
            "is_active": self.is_active,
        }


# ==========================================
# LOADER — usado por invoke_agent()
# ==========================================

async def load_tenant_config(tenant_id: str, db: AsyncSession) -> Optional["TenantConfig"]:
    """
    Carga la configuracion de un tenant desde Postgres.
    Retorna None si el tenant no tiene configuracion registrada
    (modo retrocompatible: todos los intents y tools activos).

    Ejemplo de uso en invoke_agent():
        async with AsyncSessionLocal() as db:
            tc = await load_tenant_config(tenant_id, db)
        active_intents = tc.active_intents if tc else None
    """
    result = await db.get(TenantConfig, tenant_id)
    return result
