# Product Requirements Document (PRD) - Proyecto "Laika" 🚀

## 1. Visión del Producto

**Laika** es un sistema SaaS B2B "Plug & Play" de Inteligencia Artificial Autónoma. Actúa como el cerebro cognitivo que orquesta acciones en la nube o CRMs. Su principal propuesta de valor es ser ridículamente sencillo de implementar: basta con que la empresa conecte sus cuentas o URLs base, y Laika provee servicio de "Día 1" gracias a la capacidad de ingesta agnóstica de n8n (Omnicanalidad nativa para Mail, Slack, Web y WhatsApp).

A diferencia de agentes "internos" (como Manus.ai), Laika funciona de "Control Inverso" asíncrono sobre n8n. Esto quiere decir que Laika usa herramientas de Python para pedirle directamente a n8n, vía su API, que ejecute "Workflows de Acción" completos, haciendo que n8n se comporte como sus manos digitales (El sistema nervioso).

## 2. Objetivos Principales y Filosofía

1. **El Paradigma Cognitivo vs Determinista:** _"La IA decide QUÉ hacer; Python decide CÓMO hacerlo"_. Y a diferencia de asistentes aislados, Python gatilla sistemáticamente flujos de n8n vía API REST para accionar los CRMs o ecosistemas locales del cliente ("Control Inverso").
2. **Omnicanal y Proactivo (Heartbeat):** Laika no sólo reacciona a los Webhooks recibidos, sino que ejecuta "cron-jobs" internos ("Latidos") para despertar agentes que verifiquen bases de datos asíncronamente y notifiquen al N8N de forma proactiva si hay anomalías.
3. **Aislamiento Multitenant (Data Isolation):** Todo dato, request, checkpointer de la conversación o registro RAG está atado indisolublemente a un `tenant_id`. Las búsquedas en pgvector se filtrarán a nivel base de datos preventivamente (Row-Level Security) para evitar filtración trans-empresarial de Inteligencia Artificial.
4. **Resiliencia y Observabilidad (Langfuse + DLQ):** Todo "pensamiento" de agente, consumo de token e integración documentaria, será trasteado en vivo usando el dashboard Open Source **Langfuse**. Adicionalmente, si el worker de IA agota reintentos u ocurre un timeout masivo, Python avisará a n8n usando el Patrón **Dead Letter Queue (DLQ)** para avisar al humano final y no abandonarlo con un silencio infinito.
5. **Gestión de Estado Robusta y Compleja:** LangGraph guardará cada hilo asíncrono con el Checkpointer de Postgres (`AsyncPostgresSaver`). El `LaikaState` contiene estructuras dedicadas: `current_intent`, `extracted_entities`, `plan`, `clarification_needed`, `background_task_id`, `channel` y `formatted_response`.
6. **Costo-Eficiencia Semántica:** Reducir gasto recurrente aplicando una **Caché Semántica estricta** (Vía pgvector en PostgreSQL) que intercepta queries redundantes antes del grafo de LangGraph.
7. **Local-First para Vectores:** Aprovechar GPU T-1000 (con modelos ultra-optimizados <1GB de VRAM como `MiniLM-L12`) para mantener la soberanía sobre los embeddings de forma nativa en español.
8. **SaaS Plug & Play por Tenant (`TenantConfig`):** Cada empresa puede personalizar la "personalidad" de Laika (backstory, intents permitidos, tools activas, reglas de formato por canal) sin tocar código. La configuración se persiste en `tenant_configs` y se inyecta en el grafo en tiempo de ejecución vía `RunnableConfig.configurable`.
9. **Human-in-the-Loop (Clarificación):** Cuando la intención del usuario es ambigua, el grafo se pausa con `interrupt()` de LangGraph, despacha una pregunta específica al usuario, y reanuda en el mismo checkpoint una vez que el usuario responde. Cero mensajes perdidos.
10. **Formato Adaptativo por Canal:** Una respuesta correcta semánticamente puede ser inútil si está mal formateada para el canal destino. Un `formatter_node` dedicado adapta el output al límite de caracteres y convenciones de Telegram, WhatsApp, Slack, Email o API pura antes del despacho final.

## 3. Requisitos de Hardware y Entorno

- **CPU:** Intel(R) Core(TM) i7-10850H CPU @ 2.70GHz
- **RAM:** 32 GB (31.8 GB usable)
- **GPU:** Quadro T-1000 (4GB VRAM) - _Exclusiva para Embeddings locales optimizados (Ej. MiniLM)_.
- **Despliegue:** 100% dockerizado (`docker-compose`).

## 4. Requerimientos Funcionales

### 4.1. Gateway Asíncrono y Encolamiento

- **API REST Asíncrona:** `FastAPI` con los siguientes grupos de endpoints:
  - `POST /v1/hooks/n8n` — Ingesta omnicanal principal (202 inmediato + Celery dispatch).
  - `GET /v1/jobs/{task_id}` — Estado de tareas Celery asíncronas (para tareas largas).
  - `POST/GET/PATCH/DELETE /v1/tenants` — CRUD de configuración por tenant.
  - `POST /v1/documents/ingest` — Ingesta de documentos RAG por tenant.
  - `GET /health` — Healthcheck con estado de modelos y rotación.
- **Contrato Estricto (JSON Payload):** Todo request a FastAPI debe obligatoriamente validar bajo Pydantic: `{"tenant_id": "org_x", "thread_id": "wa_y", "channel": "whatsapp", "user_query": "...", "metadata": {}}`.
- **Worker Queues y Fire-and-Forget:** FastAPI procesa JWT, Pydantic y un `202 Aceptado` inmediato. Luego delega a Background Workers, apoyados por **Redis** (Cola de mensajes estricta y control de rotación de IA).
- **Cross-Tenant Guard:** El `tenant_id` del JWT Bearer se verifica contra el `tenant_id` del payload. Discrepancias retornan `403 Forbidden` inmediatamente.
- **Rate Limiting por Tenant:** Ventana deslizante en Redis (`WEBHOOK_RATE_LIMIT_PER_MINUTE`, default 60). Excedido el límite → `429 Too Many Requests` con header `Retry-After`.

### 4.2. Workflows Agénticos (Los Patrones de Anthropic)

La topología del grafo LangGraph implementada es:

```
START → moderation → router → [casual | clarification | planner | orchestrator | task_dispatcher]
                               clarification → router  (loop hasta resolución)
                               planner → orchestrator
                               orchestrator ⇄ tool_node  (loop hasta sin tool_calls)
                               orchestrator → evaluator → [orchestrator (retry) | formatter]
                               casual → formatter
                               task_dispatcher → formatter
                               formatter → END
```

Patrones implementados:

1. **Moderation (Guardián de Seguridad):** Primer nodo del grafo. Detecta prompt injection y jailbreak. Si `safe=false` → `current_intent = "blocked"` → END inmediato. Modo fail-open: si el modelo de moderación falla, deja pasar la petición.
2. **Router con Feature Flags (El Portero del Costo):** Un LLM Tier-2 (Groq velocista) clasifica el intent en: `casual`, `cotizacion`, `investigacion_complex`, `soporte`, `tarea_larga`, o `ambiguous`. Aplica la whitelist de `active_intents` del tenant: intents no habilitados caen automáticamente a `casual`.
3. **Clarification Node (Human-in-the-Loop):** Para intents `ambiguous`, usa `langgraph.types.interrupt()` para pausar el grafo en el checkpoint de Postgres. Despacha una pregunta específica al usuario y reanuda al recibir la respuesta en el mismo `thread_id`. No pierde contexto entre el pauso y el reanudo.
4. **Agentic RAG:** La memoria documental se expone como `RAG_Search_Tool`. El agente consulta `pgvector` (filtrado por `tenant_id`), evalúa si el resultado es útil y reformula la query autónomamente si no lo es.
5. **Orchestrator-Worker (Plan & Execute):** Para `investigacion_complex`, el `planner_node` genera un plan JSON (`{"steps": [...]}`) antes del orquestador. El orquestador ejecuta `asyncio.gather` para paralelizar búsquedas RAG + web.
6. **Task Dispatcher (Tareas Largas):** Para intent `tarea_larga`, responde inmediatamente al usuario con un ACK + `task_id`, y despacha la tarea a `run_long_background_task` (Celery, time_limit=30min). Cuando termina, notifica proactivamente al usuario con el resultado.
7. **Evaluator-Optimizer (El Supervisor Final):** Revisa el borrador del orquestador. Si lo rechaza, inyecta la crítica como `AIMessage` y vuelve al orquestador (máximo 2 reintentos). Registra el veredicto como score en Langfuse.
8. **Formatter (Adaptación por Canal):** Último nodo antes de END. Lee `state["channel"]` y aplica reglas de formato: límite de caracteres, strip de Markdown para canales que no lo soportan, truncado inteligente. Reglas sobreescribibles por tenant via `channel_config`.

### 4.3. Herramientas Base (Tools) y Scripts Python

**Tools universales** (activas para todos los tenants, personalizadas únicamente por prompts):
- `perform_rag_search(query)`: Consulta autónoma a `pgvector` con reformulación LLM automática si el primer resultado es irrelevante. Tenant-isolated via `WHERE tenant_id = 'X'`.
- `web_search(query)`: Integración con Tavily para conocimiento en tiempo real. API key configurable por entorno.
- `n8n_workflow_execution(workflow_id, action_payload)`: Ordena a n8n ejecutar un workflow prefabricado vía su API REST.
- `trigger_dlq_webhook(...)`: HTTP POST de emergencia al webhook de fallo en n8n cuando todos los reintentos se agotan.

**Nodos de procesamiento** (sin costo LLM, lógica determinista pura):
- `Semantic_Cache_Check`: Compara el vector de la pregunta entrante contra `pgvector` (filtrado por `tenant_id`). Si similitud coseno >96% → respuesta instantánea sin invocar el grafo.
- `formatter_node`: Adapta la respuesta final al canal destino (Telegram, WhatsApp, Slack, Email, API). Sin llamadas LLM.
- `clarification_node`: Pausa el grafo con `interrupt()` y espera la respuesta del usuario. Sin llamadas LLM.
- `task_dispatcher_node`: Genera ACK inmediato y despacha a Celery. Sin llamadas LLM.

**Tools opcionales** (activables por tenant via `active_tools` en `TenantConfig`, con config de conexión propia):
- Ejemplo: `crm_lookup`, `invoice_generator`, `hr_leave_request` — a implementar según necesidades de cada empresa. Sus parámetros de conexión (URL, API keys) se almacenan en `tenant_config.active_tools`.

## 5. Fases de Desarrollo

### ~~FASE 1: Fundación y Resiliencia (Enterprise Core)~~ ✅ COMPLETADA

- `docker-compose.yml` con 10 servicios: `laika_api`, `celery_worker`, `celery_beat`, `redis`, `postgres`, `n8n`, `langfuse_server`, `langfuse_worker`, `langfuse_db`, `ngrok`.
- Modelo determinista de variables con `pydantic-settings` (`config.py`).
- Celery + Redis garantizando `202 Aceptado` inmediato con `task_id` en la respuesta.
- LangGraph con `AsyncPostgresSaver` namespaced (`tenant_id::thread_id`) para aislamiento de checkpoints.
- Grafo completo: `moderation → router → [casual|planner|orchestrator|clarification|task_dispatcher] → evaluator → formatter → END`.
- Structlog JSON + Langfuse v3 con scores por nodo.
- LiteLLM con rotación dinámica, cooldown Redis por modelo y fallback automático en 429.
- Cross-tenant guard en todos los endpoints (`403` si JWT tenant ≠ payload tenant).
- Rate limiting por tenant con ventana deslizante Redis.

### ~~FASE 2: Advanced Agentic RAG~~ ✅ COMPLETADA

- Orquestador Plan & Execute: `planner_node` genera JSON `{"steps": [...]}` → orquestador lo ejecuta como hoja de ruta.
- RAG Tool con reformulación LLM automática (no heurística).
- `asyncio.gather` para paralelizar RAG + web en `investigacion_complex`.
- Caché semántica (`pgvector` coseno >96%) antes del grafo — 0 costo, 0 latencia.
- Embeddings locales en GPU T-1000 (`paraphrase-multilingual-MiniLM-L12-v2`).

### ~~FASE 3: SaaS Plug & Play y Human-in-the-Loop~~ ✅ COMPLETADA

- **`TenantConfig`** — tabla `tenant_configs` con backstory, `active_intents`, `active_tools`, `channel_config` por empresa. CRUD API en `/v1/tenants`.
- **Feature flags cognitivos** — `active_intents` actúa como whitelist en el router. Intents no habilitados para el tenant caen a `casual` sin error.
- **Clarification node** — `interrupt()` de LangGraph pausa el grafo, despacha pregunta al usuario, reanuda al recibir respuesta en el mismo `thread_id`. Checkpoint persiste en Postgres.
- **Formatter node** — Adapta la respuesta a Telegram, WhatsApp, Slack, Email o API. Sobreescribible por `channel_config` del tenant.
- **Task dispatcher** — Intent `tarea_larga` genera ACK inmediato + despacha `run_long_background_task` (Celery, 30min time_limit). Notificación proactiva al usuario al terminar.

### FASE 4 (Pendiente): Extensiones Enterprise

- Onboarding automático de nuevos tenants vía n8n API.
- Dashboard de consumo de tokens/mes por tenant.
- Keys de LLM por tenant (facturación separada).
- Tools opcionales de industria (`crm_lookup`, `invoice_generator`) con config en `active_tools`.

## 6. Filosofía del Proyecto (La Regla de Oro)

> **"La IA decide _QUÉ_ hacer; Python decide _CÓMO_ hacerlo."**

Los agentes tienen autonomía para encriptar planes de ejecución, validar información y decidir cuándo iterar, pero son nuestros robustos escudos deterministas de Python los encargados de salvaguardar el performance asíncrono y proteger al usuario de fallas insuperables de la inteligencia.
