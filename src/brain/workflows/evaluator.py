import yaml
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from src.core.state import LaikaState
from src.brain.llm_proxy import get_routing_llm
from structlog import get_logger

logger = get_logger("laika_evaluator")

# Cargamos registry
with open("src/config/prompts_registry.yaml", "r", encoding="utf-8") as file:
    prompts = yaml.safe_load(file)

async def evaluator_node(state: LaikaState, config: RunnableConfig) -> dict:
    """
    Patrón 5: Evaluator-Optimizer
    Filtro de seguridad que lee el borrador generado por los Workers u Orquestador.
    Verifica Reglas de Negocio antes de dar luz verde a n8n.
    Registra el veredicto como Score en Langfuse v3.
    """
    configurable = config.get("configurable", {})
    tenant_id = configurable.get("tenant_id", "unknown")
    
    # El borrador a evaluar asumiendo que es el último mensaje (que generó la IA)
    # Limitar historial para no saturar el contexto (safety cap)
    all_messages = state["messages"]
    if len(all_messages) > 20:
        all_messages = all_messages[-20:]
    draft_message = all_messages[-1]
    
    # Solo procesar si el último mensaje fue generado por la IA
    if not isinstance(draft_message, AIMessage):
         return {}
    
    logger.info("evaluator_node_start", tenant=tenant_id)

    sys_prompt = SystemMessage(content=prompts["system_prompts"]["evaluator_node"])
    # IMPORTANTE: Groq requiere que la palabra 'json' aparezca en algún mensaje
    # cuando se usa response_format={"type": "json_object"}.
    draft_eval = HumanMessage(content=f"Evalúa este borrador y responde ÚNICAMENTE en formato JSON: {draft_message.content}")

    # Selección dinámica del modelo con rotación automática en 429
    from src.brain.rate_limiter import set_model_cooldown as _cooldown
    response = None
    for _attempt in range(3):
        llm = await get_routing_llm()
        llm_json = llm.bind(response_format={"type": "json_object"})
        _model_id = getattr(llm, "_laika_model_id", "unknown")
        try:
            # CRÍTICO: pasar `config` propaga los callbacks de Langfuse al LLM.
            response = await llm_json.ainvoke(
                [sys_prompt, draft_eval], config=config
            )
            break
        except Exception as _exc:
            _err = str(_exc).lower()
            if "429" in _err or "rate_limit" in _err or "rate limit" in _err:
                logger.warning("evaluator_429_cooldown",
                               model=_model_id, attempt=_attempt + 1)
                await _cooldown(_model_id, seconds=60)
                if _attempt < 2:
                    continue
            raise

    if response is None:
        raise RuntimeError("Todos los modelos de routing agotados en evaluator")

    try:
        data = json.loads(response.content)
        status = data.get("status", "rejected")
        reason = data.get("feedback", "")
        
        # ==========================================
        # LANGFUSE v3: Registrar Score del Evaluador
        # ==========================================
        # El CallbackHandler activo en este contexto de LangChain permite
        # obtener el trace_id del grafo actual para asociar el score.
        _register_evaluator_score(status, reason, tenant_id, config)

        if status == "rejected":
            current_retry = state.get("retry_count", 0)
            logger.warning("evaluator_rejected_draft",
                           reason=reason, tenant=tenant_id, retry_count=current_retry)

            # Inyectar la critica como HumanMessage para que el orquestador la lea
            # en el proximo intento y corrija exactamente lo que falla.
            return {
                "messages": [
                    AIMessage(
                        content=f"[CRITICA INTERNA - NO ENVIAR AL USUARIO] "
                                f"Borrador rechazado. Razon: {reason}. "
                                "Por favor genera una respuesta corregida."
                    )
                ],
                "retry_count": current_retry + 1,
                "last_eval_approved": False,
            }
        else:
            logger.info("evaluator_approved_draft", tenant=tenant_id)
            return {"last_eval_approved": True}

    except Exception as e:
        logger.error("evaluator_error_fallback", error=str(e), tenant=tenant_id)
        # Ante duda del evaluador (errores JSON), aprobamos para no bloquear el flujo
        return {"last_eval_approved": True}


def route_after_evaluator(state: LaikaState) -> str:
    """
    Borde condicional post-evaluator.
    Si el borrador fue rechazado Y aun quedan reintentos -> volver al orquestador.
    Si fue aprobado O se agotaron los reintentos -> terminar.
    """
    MAX_RETRIES = 2
    approved = state.get("last_eval_approved", True)
    retry_count = state.get("retry_count", 0)

    if not approved and retry_count <= MAX_RETRIES:
        return "retry_orchestrator"
    return "done"


def _register_evaluator_score(status: str, reason: str, tenant_id: str, config: RunnableConfig) -> None:
    """
    Registra el veredicto del evaluador como Score en Langfuse v3.

    Estrategia de obtención del trace_id (Langfuse SDK 3.14+):
      - LangGraph pasa callbacks como una lista en config["callbacks"] ANTES de
        invocar el grafo, pero dentro de los nodos LangGraph ya envuelve los callbacks
        en un AsyncCallbackManager (que no es iterable directamente).
      - Solución robusta: acceder a config["callbacks"] que puede ser tanto una lista
        como un AsyncCallbackManager, y usar getattr para evitar errores de iteración.
      - El CallbackHandler v3 expone `last_trace_id` (str | None) con el trace_id
        del último trace iniciado por este handler.

    Non-blocking: si Langfuse no está disponible, loguea y continúa.
    """
    try:
        score_value = 1.0 if status == "approved" else 0.0
        comment = "approved by evaluator" if status == "approved" else f"rejected: {reason}"

        # LangGraph puede envolver los callbacks en AsyncCallbackManager dentro del nodo.
        # Intentamos acceder a .handlers (atributo de AsyncCallbackManager) primero,
        # y si no existe asumimos que ya es una lista.
        callbacks_raw = (config or {}).get("callbacks", [])
        handlers_list = getattr(callbacks_raw, "handlers", None)
        if handlers_list is None:
            # Es una lista directa (llamada desde fuera de un nodo compilado)
            try:
                handlers_list = list(callbacks_raw)
            except TypeError:
                handlers_list = []

        langfuse_handler = next(
            (cb for cb in handlers_list if type(cb).__name__ == "LangchainCallbackHandler"),
            None,
        )

        if langfuse_handler is None:
            logger.debug("evaluator_score_skipped_no_handler", tenant=tenant_id)
            return

        # Langfuse SDK 3.14+: `last_trace_id` contiene el trace_id del último trace
        # iniciado por este handler. `trace` (objeto antiguo) ya no existe.
        trace_id = getattr(langfuse_handler, "last_trace_id", None)

        if not trace_id:
            logger.debug("evaluator_score_skipped_no_trace_id", tenant=tenant_id)
            return

        # Usar el cliente SDK singleton (no crear una nueva instancia)
        from langfuse import get_client
        lf_client = get_client()

        lf_client.create_score(
            trace_id=trace_id,
            name="evaluator_verdict",
            value=score_value,
            comment=comment,
            data_type="NUMERIC",
        )
        logger.info(
            "langfuse_score_registered",
            verdict=status,
            score=score_value,
            trace_id=trace_id,
            tenant=tenant_id,
        )

    except Exception as e:
        # Non-blocking: Langfuse falla silenciosamente para no romper el flujo del agente
        logger.warning("langfuse_score_failed", error=str(e), tenant=tenant_id)
