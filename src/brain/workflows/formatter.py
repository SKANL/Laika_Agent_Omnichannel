"""
formatter.py — Nodo de Formato Adaptativo por Canal

Transforma la respuesta final del orquestador/casual al formato óptimo
según el canal de entrega (Telegram, WhatsApp, Slack, Email, API pura).

Por qué es un nodo separado y no lógica en el orquestador:
  - Single Responsibility: el orquestador piensa, el formatter adapta.
  - Testable independientemente: puedes probar formatos sin invocar LLMs.
  - Tenant-configurable: channel_config en TenantConfig puede sobreescribir los defaults.

Flujo:
  evaluator (aprobado) → formatter → END

El formatter lee state.channel (inyectado en input_state por invoke_agent)
y aplica reglas sin llamar a ningún LLM (operación O(1), sin costo).
"""
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
import re

from src.core.state import LaikaState
from structlog import get_logger

logger = get_logger("laika_formatter")


# ==========================================
# REGLAS DE FORMATO POR CANAL
# ==========================================

_DEFAULT_CHANNEL_RULES: dict[str, dict] = {
    "telegram": {
        "max_chars": 4096,
        "use_markdown": True,
        "bold_marker": "*",
        "code_marker": "`",
        "list_marker": "•",
    },
    "whatsapp": {
        "max_chars": 4096,
        "use_markdown": True,
        "bold_marker": "*",
        "code_marker": "```",
        "list_marker": "•",
    },
    "slack": {
        "max_chars": 40000,
        "use_markdown": True,
        "bold_marker": "*",
        "code_marker": "`",
        "list_marker": "•",
    },
    "email": {
        "max_chars": 100000,
        "use_markdown": False,
        "use_html": True,
        "list_marker": "-",
    },
    "api": {
        "max_chars": 0,  # 0 = sin límite
        "use_markdown": False,
        "strip_markdown": True,
    },
    "unknown": {
        "max_chars": 2000,
        "use_markdown": False,
    },
}


def _merge_rules(channel: str, tenant_channel_config: dict | None) -> dict:
    """
    Combina las reglas default del canal con la config del tenant.
    La config del tenant tiene prioridad sobre los defaults.
    """
    defaults = _DEFAULT_CHANNEL_RULES.get(channel, _DEFAULT_CHANNEL_RULES["unknown"]).copy()
    if tenant_channel_config:
        overrides = tenant_channel_config.get(channel, {})
        defaults.update(overrides)
    return defaults


def _strip_markdown(text: str) -> str:
    """Elimina marcadores Markdown comunes para canales que no los soportan."""
    # Quitar negritas (**text** o __text__)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    # Quitar cursivas (*text* o _text_)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Quitar inline code
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Quitar headers
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    """Trunca el texto al límite del canal con sufijo informativo."""
    if max_chars and len(text) > max_chars:
        suffix = "\n\n_(Respuesta truncada por límite del canal)_"
        return text[: max_chars - len(suffix)] + suffix
    return text


def _format_for_channel(text: str, rules: dict) -> str:
    """Aplica las transformaciones de formato según las reglas del canal."""
    if rules.get("strip_markdown"):
        text = _strip_markdown(text)

    max_chars = rules.get("max_chars", 0)
    text = _truncate(text, max_chars)

    return text


async def formatter_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Formatea la respuesta final para el canal de destino.

    Lee:
      - state["channel"]            → canal de entrega ("telegram", "email", etc.)
      - state["messages"][-1]       → último AIMessage con la respuesta raw
      - config.configurable["channel_config"] → overrides de formato del tenant

    Escribe:
      - state["formatted_response"] → texto listo para enviar al usuario
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")

    # Canal: primero el state (inyectado por invoke_agent), fallback a configurable
    channel = state.get("channel") or configurable.get("channel", "unknown")
    channel = (channel or "unknown").lower()

    # Config de canal del tenant (puede ser None si el tenant no tiene overrides)
    tenant_channel_config: dict | None = configurable.get("channel_config")

    # Extraer el último AIMessage
    all_messages = state.get("messages", [])
    raw_response = ""
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage):
            raw_response = msg.content
            break

    if not raw_response:
        logger.warning("formatter_no_ai_message", tenant=tenant_id)
        return {"formatted_response": "No se pudo generar una respuesta."}

    # Obtener reglas combinadas (default + tenant override)
    rules = _merge_rules(channel, tenant_channel_config)

    # Aplicar formato
    formatted = _format_for_channel(raw_response, rules)

    logger.info(
        "formatter_applied",
        tenant=tenant_id,
        channel=channel,
        raw_len=len(raw_response),
        formatted_len=len(formatted),
    )

    return {"formatted_response": formatted}
