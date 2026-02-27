# ==========================================
# PLANNER NODE: PLAN & EXECUTE (Patrón Anthropic)
# ==========================================
# Para intenciones de tipo 'investigacion_complex', el planner
# descompone la consulta del usuario en sub-tareas estructuradas
# antes de pasarlas al orquestador.
#
# Flujo: router → planner → orchestrator
# El plan se deposita en state['plan'] para que el orquestador
# lo use como hoja de ruta al dirigir sus herramientas.

import json
import os

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.brain.rate_limiter import set_model_cooldown
from src.core.state import LaikaState
from src.brain.llm_proxy import get_routing_llm
from structlog import get_logger

logger = get_logger("laika_planner")

_PROMPTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "prompts_registry.yaml"
)
with open(_PROMPTS_PATH, "r", encoding="utf-8") as _f:
    _prompts = yaml.safe_load(_f)


async def planner_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patrón Anthropic: Orchestrator-Worker — fase de planificación.

    Genera una lista de sub-tareas concretas que el orquestador ejecutará
    usando sus herramientas (RAG, web_search, n8n). Esto garantiza que:
      1. El LLM pesado recibe instrucciones precisas en lugar de preguntas abiertas.
      2. Cada sub-tarea puede controlarse y loguearse individualmente.
      3. Los reintentos del evaluador son más quirúrgicos (se re-ejecuta el paso fallido).

    Si el planner falla, devuelve un plan vacío y el orquestador opera en modo reactivo.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")

    last_message = state["messages"][-1]
    logger.info("planner_node_start", tenant=tenant_id)

    # Prompt del planner SIN backstory global — el planner es un descomponedor
    # de tareas puro. El backstory de "Laika" solo agrega ruido aquí.
    planner_prompt = _prompts.get("system_prompts", {}).get("planner_node", "")
    sys_msg = SystemMessage(content=planner_prompt)
    user_msg = HumanMessage(content=last_message.content)

    response = None
    for _attempt in range(3):
        llm = await get_routing_llm()
        llm_json = llm.bind(response_format={"type": "json_object"})
        _mid = getattr(llm, "_laika_model_id", "unknown")
        try:
            response = await llm_json.ainvoke([sys_msg, user_msg], config=config)
            break
        except Exception as exc:
            _err = str(exc).lower()
            if "429" in _err or "rate_limit" in _err:
                await set_model_cooldown(_mid, seconds=60)
                if _attempt < 2:
                    continue
            raise

    if response is None:
        logger.warning("planner_fallback_empty_plan", tenant=tenant_id)
        return {"plan": []}

    try:
        data = json.loads(response.content)
        plan = data.get("steps", data.get("plan", []))
        if isinstance(plan, str):
            plan = [plan]
        elif not isinstance(plan, list):
            plan = []
        logger.info("plan_generated", tenant=tenant_id, steps=len(plan))
        return {"plan": plan}
    except Exception as e:
        logger.error("planner_json_parse_failed", error=str(e), tenant=tenant_id)
        return {"plan": []}
