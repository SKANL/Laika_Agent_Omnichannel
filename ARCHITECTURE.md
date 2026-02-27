# Documento de Arquitectura y Decisiones Técnicas - Proyecto "Laika" 🏛️

Este documento actúa como la única fuente de verdad técnica para el desarrollo del proyecto "Laika". Detalla las decisiones de diseño, la estructura rigurosa del stack elegido y el flujo lógico basado en patrones de grado de producción ("Enterprise-grade").

## 1. El Core Cognitivo vs. Determinista (El Escudo de Python)

Para que Laika sea sustentable y no gaste inútilmente "tokens" de Free Tiers, los nodos de LangGraph y el LLM funcionan exclusivamente encapsulados detrás de **Cientos de Scripts de Python (El Escudo)**.

El modelo a seguir (Aconsejado como la mejor práctica de la industria) es:

1. **Agente IA (Razonamiento / Motor Cognitivo):** Decide si la urgencia es de un nivel 1 a 5, escoge qué herramienta de Python usar, deduce intenciones complejas, y redacta de manera natural generando outputs estructurados (JSONs validados).
2. **Scripts Python (Las Manos Deterministas):** Python maneja bases de datos (SQL), cálculos matemáticos de fechas, validación de variables, formateo de payloads finales para APIs, y llamadas a la API de N8N. NUNCA la IA construye el payload final externo directamente. Esto garantiza velocidad milisegundo a costo 0$, y evita "alucinaciones" críticas.

### 1.1. Agentic RAG vs. Vanilla RAG (Nuestra Filosofía de Memoria)

Laika no inyecta bases de datos "a ciegas" en el prompt (Vanilla). Implementamos **Agentic RAG**:

- Los vectores almacenados en PostgreSQL (pgvector) no entran a la fuerza al flujo.
- La base de datos vectorial se expone como una **Herramienta (Tool)** dentro del Grafo.
- El Agente **decide autónomamente** formular la query, consultar la memoria, leer los documentos devueltos y, si no son útiles, **reformula** la búsqueda hasta obtener el dato exacto antes de emitir una respuesta.

> [!WARNING]
> **Veto Temporal a Knowledge Graphs (Grafos de Conocimiento / Neo4j):**
> Excluido en la Fase Base. Su indexación exhaustiva destruiría los "Rate Limits" de los proveedores gratuitos (Groq, Cerebras). Queda como _Add-on Enterprise_ a futuro.

## 2. Stack Tecnológico Aprobado

### 2.1. Gateway HTTP y Tareas en Segundo Plano (Async Workers)

- **Motor Principal:** Python 3.12+
- **Framework Web:** FastAPI (API Endpoints).
- **Servidores WSGI/ASGI:** `Uvicorn` + `Gunicorn`.
- **Encolamiento y Broker:** Redis. **Exclusivamente** como Message Broker de Celery/RQ o caché llave-valor clásica.
- **Validación Bidireccional (Contratos Pydantic):** Laika y n8n comparten un molde inquebrantable de datos donde exigen obligatoriamente la metadata `tenant_id` y `thread_id` en las intercepciones HTTP iniciales. Si el JSON entrante desde n8n falla o el output de la IA no es un JSON estructurado, Pydantic captura y exige reintento.
- **Autenticación:** `python-jose[cryptography]` vía JWT tokens.
- **Observabilidad Dual (`structlog` + Langfuse):** System logging tradicional se rige por estructuración JSON estricta (Structlog). La "caja negra" cognitiva de los LLMs, tiempos asíncronos de nodos de LangGraph y contexto de inferencias enviadas serán traceadas y parseadas a dashboards gráficos usando **Langfuse (Open Source)**.

### 2.2. Capa de Inteligencia Activa ("Brain") y Estrategia de Modelos

Para soportar los niveles gratuitos (Free Tiers) sin colapsar, no existirán modelos "hardcodeados" en el código. Implementamos un ecosistema guiado por registros y `LiteLLM`:

- **Categorización por Roles (`models_registry.yaml`):**
  - **Tier 1 (Heavy Lifters):** Modelos de gran contexto (ej. Cerebras `qwen-3-235b-a22b`, 60K TPM) para el **Orquestador** y **Agentic RAG**.
  - **Tier 2 (Velocistas/Routers):** Modelos ultrarrápidos (ej. Groq `llama-3.1-8b-instant`, <200ms) para clasificar intenciones iniciales y ahorrar tokens en trivialidades.
- **Enrutamiento Dinámico y Rate Limits:** `LiteLLM` manejará "Fallbacks" (si Groq da error 429, salta inmediatamente a Cerebras) y "Load Balancing". Además, un monitor en Redis trackeará proactivamente el consumo TPM/RPM bloqueando llamadas al proveedor si está cercano al límite.
- **Gestión de Prompts:** Todo texto estricto (Reglas, Backstory) residirá disociado de Python en un `prompts_registry.yaml`.

Adicionalmente, usaremos los **5 Patrones de Workflows Agénticos** de Anthropic:

- **Enrutamiento (Routing):** Un nodo ligero clasifica la entrada (Charla vs. Tarea Compleja).
- **Orquestador-Trabajador:** Para tareas que requieren investigación multifase.
- **Cadena de Prompts (Prompt Chaining):** Dividir tareas colosales en sub-llamadas secuenciales pasando solo el contexto vital ("Prompt Injection" dinámica).
- **Paralelización:** Lanzar búsquedas web y RAG simultáneas.
- **Evaluador-Optimizador:** Supervisor final que lee el borrador, verifica "Reglas de Negocio" y aprueba o rechaza el envío a n8n.

### 2.3. Persistencia de Estado y Memoria ("Checkpointing")

- **`LaikaState` (TypedDict completo):** El state que viaja entre nodos contiene:
  - `messages` — historial de la conversación (`add_messages` reducer, nunca sobreescribe).
  - `current_intent` — clasificación del router (`casual`, `cotizacion`, `investigacion_complex`, `soporte`, `tarea_larga`, `ambiguous`, `blocked`).
  - `extracted_entities` — entidades extraídas por el orquestador/RAG (incluye `clarification_question` cuando intent es `ambiguous`).
  - `worker_errors` — trazas de errores de tools para el DLQ pattern.
  - `retry_count` / `last_eval_approved` — control del loop evaluador-optimizador (máx. 2 reintentos).
  - `plan` — lista de sub-objetivos generada por `planner_node` para `investigacion_complex`.
  - `clarification_needed` — flag que activa el enrutamiento hacia `clarification_node`.
  - `background_task_id` — ID de la tarea Celery para intents `tarea_larga` (consultable en `/v1/jobs/{task_id}`).
  - `channel` — canal de entrega (`telegram`, `whatsapp`, `slack`, `email`, `api`). Usado por `formatter_node`.
  - `formatted_response` — respuesta final procesada por `formatter_node`. Es el dato que `invoke_agent` despacha a n8n.
- **Namespacing de threads:** El `thread_id` en el checkpointer viaja como `"{tenant_id}::{thread_id}"`. Dos tenants con el mismo `thread_id` jamás comparten estado.
- **Resumability con `interrupt()`:** `AsyncPostgresSaver` persiste el estado en el punto exacto del `interrupt()`. Al llegar el siguiente mensaje en el mismo `thread_id`, `invoke_agent` detecta el checkpoint pausado via `aget_state()` y reanuda con `Command(resume=user_response)` en lugar de crear un nuevo estado.

### 2.4. Capa de Recuperación Vectores, RAG y Multitenant (RLS)

- **Multitenancy y Data Isolation (Seguridad B2B):** Es **imposible** ejecutar una similitud de vectores globales. Cada chunk inyectado a la base de datos contará con la columna `tenant_id`. Todas las búsquedas vectoriales del Agentic RAG añadirán obligatoriamente el `WHERE tenant_id = 'X'` apalancando preferentemente _Row Level Security_ nativa de Postgres.
- **Embeddings Locales:** `SentenceTransformers` (`paraphrase-multilingual-MiniLM-L12-v2` o variantes cuantizadas <1GB VRAM) en la GPU Quadro T-1000.
- **Caché Semántica Exacta:** Antes de despertar al LLM, Python compara el vector de la pregunta entrante con los de PostgreSQL (`pgvector`), condicionado **siempre** al `tenant_id`. Si la similitud coseno > 96%, devuelve la respuesta pre-almacenada (0 costo, 0 latencia).

### 2.5. Resiliencia: Dead Letter Queue (DLQ) y Falla Silenciosa

- **Mecanismo de Webhook de Fallo:** Si Groq agota sus tokens, Cerebras da 502, y el Async Worker de Python se queda sin reintentos, el Worker ejecutará un HTTP POST a un `error_webhook` en n8n.
- Esto permite que n8n avise al usuario final: _"Ocurrió un contratiempo técnico procesando tus datos. Reintentaremos pronto."_ evitando "Timeouts" infinitos silenciosos.
- **Notificación proactiva en tareas largas:** Si `run_long_background_task` falla tras `max_retries=2`, envía automáticamente un mensaje de error al usuario vía `_dispatch_reply` antes de re-lanzar la excepción. El usuario nunca queda en silencio.

### 2.6. Medio Físico de Acción y "Control Inverso"

- **N8N (El Sistema Nervioso Central):** No solo envía peticiones iniciales; Laika utiliza la **API REST de n8n** para ejecutar "Workflows" prefabricados. En vez de programar una integración compleja con un ERP en Python, el agente IA le dice a Python "Ejecuta el workflow de Alta de Cliente #44", y Python dispara vía API a n8n para que este corra la integración visual como si fueran las manos del agente.
- **Omnicanalidad y Normalización Base:** N8n intercepta WhatsApp, Slack, o Webchats, los normaliza en un JSON unificado (ej. `{canal: "wa", id: "123", texto: "Hola"}`), y se los entrega transparentemente al Gateway Laika para que opere agnóstico a la plataforma.
- **Latido Proactivo (Heartbeat):** Tareas periódicas ("cron jobs" en Python/Celery) despertarán al agente de manera autónoma (ej. a las 8:00 AM) para revisar bases de datos o métricas de ventas y gatillar alertas usando el Orquestador y enviándolas proactivamente vía N8n.

### 2.7. SaaS Plug & Play: `TenantConfig`

Cada empresa cliente tiene una fila en `tenant_configs` que define la personalidad y capacidades de Laika para esa empresa. Se carga al inicio de cada `invoke_agent()` e inyecta en `RunnableConfig.configurable` (nunca en el state visible al LLM).

| Campo | Tipo | Efecto |
|---|---|---|
| `backstory_override` | `TEXT` | Reemplaza `global_backstory` en todos los nodos |
| `active_intents` | `JSON[]` | Whitelist de intents; los demás caen a `casual` |
| `active_tools` | `JSON{}` | Tools opcionales + config de conexión por tool |
| `channel_config` | `JSON{}` | Override de reglas de formato por canal |
| `is_active` | `BOOL` | Soft-delete; false bloquea todos los mensajes del tenant |

**Principio "Tools Agnósticas":** Las tools universales (RAG, web_search, n8n) son iguales para todos los tenants. Su comportamiento se personaliza únicamente vía `backstory_override` y prompts. Las tools específicas de industria (CRM, ERP) se habilitan por tenant en `active_tools` con su propia config de conexión.

## 3. Topología de Carpetas ("Domain Driven Design" Modular)

Estructura diseñada para permitir iteración rápida y soporte en producción.

```text
c:/code/strop/strop_test_SaaS/Laika/
├── docker-compose.yml        # 10 servicios: laika_api, celery_worker, celery_beat, redis,
│                            #   postgres, n8n, langfuse_server, langfuse_worker, langfuse_db, ngrok
├── .env.local                # Secretos (git-ignored): API keys, DB password, JWT secret
├── PRD.md                    # Requerimientos Funcionales
├── ARCHITECTURE.md           # Decisiones Técnicas (este documento)
├── requirements.txt          # Dependencias Python pinneadas
└── src/
    ├── main.py                 # FastAPI lifespan + registro de routers
    │
    ├── api/
    │   ├── routers/
    │   │   ├── webhook.py      # POST /v1/hooks/n8n (cross-tenant guard, rate limit)
    │   │   ├── documents.py    # POST /v1/documents/ingest (RAG ingestion)
    │   │   ├── tenants.py      # CRUD /v1/tenants (TenantConfig)
    │   │   ├── jobs.py         # GET /v1/jobs/{task_id} (estado Celery)
    │   │   └── health.py       # GET /health, /health/models, /health/rotation
    │   └── schemas/
    │       └── requests.py     # N8NWebhookPayload, N8NAcceptanceResponse (con task_id)
    │
    ├── core/
    │   ├── config.py           # pydantic-settings: env_file tuple, CORS, rate limits, JWT
    │   ├── db.py               # Engine async + Base + RAGDocument + SemanticCache + init_db()
    │   ├── state.py            # LaikaState TypedDict (11 campos)
    │   ├── tenant_config.py    # TenantConfig SQLAlchemy model + load_tenant_config()
    │   ├── tenant_ratelimit.py # Redis sliding window rate limiter per tenant
    │   ├── security.py         # JWT verify_token + generate_dev_token
    │   └── logging_setup.py    # Structlog JSON config
    │
    ├── brain/
    │   ├── workflows/
    │   │   ├── main_graph.py       # Ensamblado del grafo + invoke_agent() + _dispatch_reply()
    │   │   ├── moderation.py       # Nodo de moderación, fail-open
    │   │   ├── router.py           # Clasificación de intents + feature flags
    │   │   ├── clarification.py    # Human-in-the-Loop con interrupt()
    │   │   ├── planner.py          # Plan & Execute para investigacion_complex
    │   │   ├── orchestrator.py     # Orquestador principal + asyncio.gather RAG+web
    │   │   ├── evaluator.py        # Evaluador-Optimizador + Langfuse scores
    │   │   ├── casual.py           # Nodo de respuesta rápida para intents triviales
    │   │   ├── task_dispatcher.py  # Despacho de tareas largas a Celery
    │   │   └── formatter.py        # Formato adaptativo por canal (sin LLM)
    │   ├── tools/
    │   │   ├── rag_tool.py         # Agentic RAG + reformulación LLM
    │   │   ├── n8n_tool.py         # Disparo de Workflows vía API N8N + DLQ trigger
    │   │   ├── web_search_tool.py  # Tavily search tool
    │   │   └── cache.py            # Caché semántica pgvector (check + store)
    │   ├── llm_proxy.py            # LiteLLM routing + get_heavy_llm() + get_routing_llm()
    │   ├── rate_limiter.py         # Cooldown Redis por modelo (post-429)
    │   ├── embeddings.py           # SentenceTransformer singleton (GPU T-1000)
    │   └── config/
    │       ├── models_registry.yaml    # Tier 1 (Heavy) + Tier 2 (Velocistas) + fallbacks
    │       └── prompts_registry.yaml   # global_backstory + todos los system prompts
    │
    └── worker/
        ├── celery_app.py           # Celery app + beat_schedule + worker_process_init signal
        └── tasks.py               # process_agentic_workflow_celery
                                    # proactive_heartbeat_trigger
                                    # run_long_background_task (time_limit=30min)
```

## 4. El Ciclo de Vida del Agente Autónomo (Recepción -> Despacho)

Viaje asíncrono con integración de Patrones Agénticos y Checkpointing:

1. **Ingesta:** n8n recibe un mensaje (con `thread_id`) y lanza un Webhook al Gateway Laika.
2. **Aceptación Asíncrona:** FastAPI valida JWT (cross-tenant guard), aplica rate limit por tenant, responde inmediatamente "202 Aceptado" con `task_id`, y despacha a Celery.
3. **Caché Semántica:** El Worker vectoriza en GPU local. Si `pgvector` dice >96% coincidencia, envía la respuesta cacheada al webhook N8N final y termina (0 tokens consumidos).
4. **Carga de `TenantConfig`:** `invoke_agent()` carga la configuración del tenant desde `tenant_configs` e inyecta `active_intents`, `backstory_override` y `channel_config` en `RunnableConfig.configurable`.
5. **Detección de Interrupt Pendiente:** `invoke_agent()` llama `aget_state(config)` para saber si el hilo tiene un `interrupt()` activo (clarificación en curso). Si hay uno → reanuda con `Command(resume=payload_msg)`. Si no → invoca con `input_state` fresco.
6. **Moderación:** `moderation_node` verifica prompt injection/jailbreak. Si `safe=false` → `blocked` → `formatter` → END.
7. **Router con Feature Flags:** `router_node` clasifica el intent. Filtra contra `active_intents` del tenant. Enruta a: `casual`, `clarification`, `planner`, `orchestrator`, o `task_dispatcher`.
8. **Human-in-the-Loop (si `ambiguous`):** `clarification_node` llama `interrupt({"question": "..."})`. El grafo se pausa. `invoke_agent()` detecta el `__interrupt__` en el resultado y despacha la pregunta al usuario. El próximo mensaje en el mismo thread reanuda el grafo.
9. **Plan & Execute / Agentic RAG (si complejo):** `planner_node` genera sub-objetivos. El orquestador los ejecuta vía `asyncio.gather` (RAG + web en paralelo).
10. **Evaluador-Optimizador:** Verifica reglas de negocio. Hasta 2 reintentos con crítica inyectada al orquestador. Registra score en Langfuse.
11. **Formatter:** `formatter_node` adapta la respuesta al canal del tenant: límite de caracteres, Markdown condicional, truncado inteligente. El resultado va a `state["formatted_response"]`.
12. **Despacho / Fallo (DLQ):** `_dispatch_reply()` envía `formatted_response` al webhook de n8n. Si falla, `trigger_dlq_webhook()` asegura que el usuario reciba una notificación de error.

FIN DEL DOCUMENTO TÉCNICO.
