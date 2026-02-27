"""
deterministic_tools.py — Herramientas Deterministas (El Escudo)

Implementa el patrón arquitectural del ARCHITECTURE.md:
  "La IA decide QUÉ hacer. Python decide CÓMO hacerlo."

Estas herramientas ejecutan operaciones que los LLMs realizan mal:
  - Aritmética precisa (LLMs cometen errores matemáticos documentados)
  - Zona horaria y fecha/hora actual (LLMs no saben la fecha real)
  - Compresión/resumen de contexto para evitar "Context Rot"

Al estar expuestas como @tool, el orquestador puede llamarlas autónomamente
cuando detecta que necesita estos cálculos. Python los ejecuta de forma
determinista — costo $0, latencia <1ms, sin riesgo de alucinación.
"""

import ast
import operator
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from structlog import get_logger

logger = get_logger("laika_deterministic_tools")

# ---------------------------------------------------------------------------
# OPERADORES SEGUROS para el evaluador de expresiones
# Previene ejecución de código arbitrario (sandbox matemático)
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_MAX_RESULT_MAGNITUDE = 1e18  # Prevenir overflow


def _safe_eval(node):
    """Evalúa un nodo AST de forma segura — solo operaciones matemáticas."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Tipo no numérico: {type(node.value)}")
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Operador no permitido: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        result = _SAFE_OPS[op_type](left, right)
        if abs(result) > _MAX_RESULT_MAGNITUDE:
            raise ValueError("Resultado excede el límite permitido (1e18)")
        return result
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Operador unario no permitido: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))
    else:
        raise ValueError(f"Nodo AST no permitido: {type(node).__name__}")


# ============================================================================
# HERRAMIENTA 1: CALCULADORA DETERMINISTA
# ============================================================================

@tool
def calculate(expression: str) -> str:
    """
    Evalúa una expresión matemática de forma determinista y sin riesgo de alucinación.
    Soporta: +, -, *, /, %, //, ** (potencia). Paréntesis permitidos.

    USA ESTA HERRAMIENTA para CUALQUIER cálculo numérico — los LLMs cometen
    errores aritméticos. Python garantiza precisión.

    Args:
        expression: Expresión matemática como string.
                    Ejemplos: "1500 * 0.16", "(3500 - 1200) / 12", "2 ** 10"
                    NUNCA incluyas variables, texto ni funciones — solo números y operadores.

    Returns:
        El resultado numérico como string, o un mensaje de error descriptivo.

    Ejemplos de uso:
        calculate("1500 * 0.16")          → "240.0"
        calculate("(3500 - 1200) / 12")  → "191.66666666666666"
        calculate("2 ** 10")              → "1024"
    """
    logger.info("calculate_called", expression=expression[:80])

    # Sanitizar: solo permitir dígitos, operadores básicos, puntos, paréntesis y espacios
    if not re.match(r'^[\d\s\+\-\*\/\%\(\)\.\*\*]+$', expression.replace("**", "^^")):
        return f"Error: Expresión inválida. Solo se permiten números y operadores (+, -, *, /, %, //, **)."

    try:
        tree = ast.parse(expression.strip(), mode='eval')
        result = _safe_eval(tree.body)
        logger.info("calculate_success", expression=expression[:40], result=result)
        return str(result)
    except ZeroDivisionError:
        return "Error: División por cero."
    except ValueError as e:
        return f"Error de validación: {e}"
    except SyntaxError:
        return f"Error: Sintaxis inválida en la expresión '{expression}'."
    except Exception as e:
        logger.error("calculate_failed", error=str(e))
        return f"Error al evaluar la expresión: {e}"


# ============================================================================
# HERRAMIENTA 2: FECHA Y HORA ACTUAL
# ============================================================================

@tool
def get_current_datetime(timezone_name: str = "America/Mexico_City") -> str:
    """
    Devuelve la fecha y hora actual del servidor en la zona horaria especificada.

    USA ESTA HERRAMIENTA cuando necesites la fecha o la hora actual.
    NUNCA asumas la fecha — los LLMs no saben la fecha real y pueden alucinar.

    Args:
        timezone_name: Nombre de zona horaria IANA. Default: "America/Mexico_City".
                       Ejemplos: "America/New_York", "Europe/Madrid", "UTC", "America/Bogota".

    Returns:
        String con fecha y hora en formato ISO 8601 más legible.

    Ejemplos de uso:
        get_current_datetime()                    → "2026-02-27 14:30:00 CST (America/Mexico_City)"
        get_current_datetime("UTC")              → "2026-02-27 20:30:00 UTC"
        get_current_datetime("Europe/Madrid")   → "2026-02-27 21:30:00 CET (Europe/Madrid)"
    """
    logger.info("get_current_datetime_called", timezone=timezone_name)

    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning("invalid_timezone", timezone=timezone_name)
        tz = ZoneInfo("UTC")
        timezone_name = "UTC (fallback — zona horaria inválida)"

    now = datetime.now(tz=tz)
    formatted = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    result = f"{formatted} ({timezone_name})"
    logger.info("get_current_datetime_success", result=result)
    return result


# ============================================================================
# HERRAMIENTA 3: EXTRACCIÓN DE ENTIDADES ESTRUCTURADAS
# ============================================================================

@tool
async def extract_entities(
    text: str,
    entity_types: str = "fecha,monto,id_referencia,nombre_empresa,nombre_persona",
    config: RunnableConfig = None,
) -> str:
    """
    Extrae entidades estructuradas de un texto libre usando el LLM ligero.
    Deposita los resultados en el estado del agente para uso posterior.

    USA ESTA HERRAMIENTA cuando el usuario menciona datos específicos (fechas, montos,
    IDs, nombres) que necesitarás usar en pasos posteriores de la conversación.
    Esto previene que el orquestador "recuerde mal" datos del historial.

    Args:
        text: Texto del cual extraer entidades.
        entity_types: Tipos de entidades separados por coma a buscar.
                      Default: "fecha,monto,id_referencia,nombre_empresa,nombre_persona"

    Returns:
        JSON string con las entidades encontradas y sus valores.

    Ejemplos de uso:
        extract_entities("La orden #4521 del 15 de enero por $3,500 está pendiente")
        → '{"id_referencia": "#4521", "fecha": "2026-01-15", "monto": "3500"}'
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.brain.llm_proxy import get_routing_llm

    logger.info("extract_entities_called", text_preview=text[:60])

    entity_list = [e.strip() for e in entity_types.split(",")]
    entity_schema = ", ".join(f'"{e}": "valor o null"' for e in entity_list)

    sys_msg = SystemMessage(
        content=(
            "Eres un extractor de entidades. Lee el texto y extrae las entidades solicitadas. "
            "Responde ÚNICAMENTE en JSON válido con las claves especificadas. "
            "Usa null si una entidad no aparece en el texto. Sin texto adicional."
        )
    )
    user_msg = HumanMessage(
        content=f"Extrae estas entidades del texto: {{{entity_schema}}}\n\nTexto: {text}"
    )

    try:
        llm = await get_routing_llm()
        llm_json = llm.bind(response_format={"type": "json_object"})
        response = await llm_json.ainvoke(
            [sys_msg, user_msg],
            config=config,
        )
        entities = json.loads(response.content)
        # Filtrar nulls para devolver solo lo encontrado
        found = {k: v for k, v in entities.items() if v is not None}
        logger.info("extract_entities_success", found=list(found.keys()))
        return json.dumps(found, ensure_ascii=False)
    except Exception as e:
        logger.error("extract_entities_failed", error=str(e))
        return f"Error al extraer entidades: {e}"
