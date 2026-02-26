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

- **State Management Definido (`LaikaState`):** El Grafo no es un simple bot conversacional. La memoria volátil compartida se instanciará con propiedades explícitas (ej. `messages` para el Thread asíncrono, `current_intent` analizado por el Router, variables `extracted_entities`, y logs de `worker_errors`).
- **Resumability:** Todo el historial asociado vive en **PostgreSQL** mediante `langgraph-checkpoint-postgres` y recupera mágicamente la memoria insertando `worker_id` y `tenant_id` en las columnas Metadata de la Persistencia de Postgres para separar los datos B2B.

### 2.4. Capa de Recuperación Vectores, RAG y Multitenant (RLS)

- **Multitenancy y Data Isolation (Seguridad B2B):** Es **imposible** ejecutar una similitud de vectores globales. Cada chunk inyectado a la base de datos contará con la columna `tenant_id`. Todas las búsquedas vectoriales del Agentic RAG añadirán obligatoriamente el `WHERE tenant_id = 'X'` apalancando preferentemente _Row Level Security_ nativa de Postgres.
- **Embeddings Locales:** `SentenceTransformers` (`paraphrase-multilingual-MiniLM-L12-v2` o variantes cuantizadas <1GB VRAM) en la GPU Quadro T-1000.
- **Caché Semántica Exacta:** Antes de despertar al LLM, Python compara el vector de la pregunta entrante con los de PostgreSQL (`pgvector`), condicionado **siempre** al `tenant_id`. Si la similitud coseno > 96%, devuelve la respuesta pre-almacenada (0 costo, 0 latencia).

### 2.5. Resiliencia: Deal Letter Queue (DLQ) y Falla Silenciosa

- **Mecanismo de Webhook de Fallo:** Si Groq agota sus tokens, Cerebras da 502, y el Async Worker de Python se queda sin reintentos, el Worker ejecutará un HTTP POST a un `error_webhook` en n8n.
- Esto permite que n8n avise al usuario final: _"Ocurrió un contratiempo técnico procesando tus datos. Reintentaremos pronto."_ evitando "Timeouts" infinitos silenciosos.

### 2.6. Medio Físico de Acción y "Control Inverso"

- **N8N (El Sistema Nervioso Central):** No solo envía peticiones iniciales; Laika utiliza la **API REST de n8n** para ejecutar "Workflows" prefabricados. En vez de programar una integración compleja con un ERP en Python, el agente IA le dice a Python "Ejecuta el workflow de Alta de Cliente #44", y Python dispara vía API a n8n para que este corra la integración visual como si fueran las manos del agente.
- **Omnicanalidad y Normalización Base:** N8n intercepta WhatsApp, Slack, o Webchats, los normaliza en un JSON unificado (ej. `{canal: "wa", id: "123", texto: "Hola"}`), y se los entrega transparentemente al Gateway Laika para que opere agnóstico a la plataforma.
- **Latido Proactivo (Heartbeat):** Tareas periódicas ("cron jobs" en Python/Celery) despertarán al agente de manera autónoma (ej. a las 8:00 AM) para revisar bases de datos o métricas de ventas y gatillar alertas usando el Orquestador y enviándolas proactivamente vía N8n.

## 3. Topología de Carpetas ("Domain Driven Design" Modular)

Estructura diseñada para permitir iteración rápida y soporte en producción.

```text
c:/code/strop/strop_test_SaaS/strop_AI_Brain_V2/
├── docker-compose.yml       # Setup `docker-compose.yml` con 7 servicios: `FastAPI`, `Celery Worker`, `Redis`, `Postgres/pgvector`, `n8n`, `Langfuse`, `Ngrok`.
├── .env                     # Tokens, BD pass
├── PRD.md                   # Requerimientos Funcionales y Patrones de Diseño
├── ARCHITECTURE.md          # Decisiones Técnicas
├── n8n_data/                # Volumen montado para n8n persistencia visual
└── src/                     # Código Laika
    ├── main.py              # Punto de entrada FastAPI
    │
    ├── api/                 # CAPA GATEWAY
    │   ├── routers/         # /webhook/n8n, /status
    │   └── schemas/         # DTOs
    │
    ├── core/                # LAYER DETERMINISTA
    │   ├── config.py
    │   ├── db.py            # Adaptadores Postgres/pgvector
    │   ├── state.py         # LaikaState (TypedDict + add_messages)
    │   └── logging_setup.py # Structlog JSON config
    │
    ├── brain/               # LAYER COGNITIVO
    │   ├── workflows/       # Patrones Anthropic: router.py, orchestrator.py
    │   ├── tools/           # "LOS SCRIPTS ESCUDO"
    │   │   ├── rag_tool.py  # Agentic RAG
    │   │   ├── n8n_tool.py  # Disparo de Workflows vía API N8N y DLQ
    │   │   └── cache.py     # Lógica pgvector Semantic-caching
    │   └── config/          # Registros YAML (Sin Hardcoding)
    │       ├── models_registry.yaml   # Configuración, rotación y fallbacks (LiteLLM)
    │       └── prompts_registry.yaml  # Configuración de contextos e inyecciones
```

## 4. El Ciclo de Vida del Agente Autónomo (Recepción -> Despacho)

Viaje asíncrono con integración de Patrones Agénticos y Checkpointing:

1. **Ingesta:** n8n recibe un mensaje (con `thread_id`) y lanza un Webhook al Gateway Laika.
2. **Aceptación Asíncrona:** FastAPI valida JWT, responde inmediatamente "202 Aceptado" a n8n, y manda el job a Redis (Worker Queue).
3. **Caché Semántica:** El Worker vectoriza en GPU local. Si `pgvector` dice >96% coincidencia de respuesta, envía la respuesta cacheada al webhook N8N final y termina.
4. **Router LLM (Patrón 1):** Un modelo rápido evalúa intención. Si es trivial, responde directo. Si es complejo, despierta el Grafo.
5. **Carga de Estado:** LangGraph usa `AsyncPostgresSaver` para recuperar la memoria de ese `thread_id`.
6. **Orquestador-Worker (Patrón 4):** Si la tarea requiere investigación, el Orquestador la subdivide en sub-tareas ("Pasos").
7. **Agentic RAG:** Los Workers usan la `RAG_Search_Tool` y `Web_Search_Tool` de manera asíncrona (Paralelización) y depositan los hallazgos crudos directo en la Base de Estado.
8. **Artífice & Evaluador (Patrón 5):** Usando inyección dinámica de prompts, el Artífice redacta. El Evaluador lo califica. Si aprueba, ordena envío.
9. **Despacho / Fallo (DLQ):** Python empaqueta los datos validados y ejecuta REST API hacia el Webhook de Recepción de n8n. (Si hubiese colapsado el worker antes, manda error al Webhook de Fallo).

FIN DEL DOCUMENTO TÉCNICO.
