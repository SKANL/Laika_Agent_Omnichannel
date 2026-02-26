# Product Requirements Document (PRD) - Proyecto "Laika" 🚀

## 1. Visión del Producto

**Laika** es un sistema SaaS B2B "Plug & Play" de Inteligencia Artificial Autónoma. Actúa como el cerebro cognitivo que orquesta acciones en la nube o CRMs. Su principal propuesta de valor es ser ridículamente sencillo de implementar: basta con que la empresa conecte sus cuentas o URLs base, y Laika provee servicio de "Día 1" gracias a la capacidad de ingesta agnóstica de n8n (Omnicanalidad nativa para Mail, Slack, Web y WhatsApp).

A diferencia de agentes "internos" (como Manus.ai), Laika funciona de "Control Inverso" asíncrono sobre n8n. Esto quiere decir que Laika usa herramientas de Python para pedirle directamente a n8n, vía su API, que ejecute "Workflows de Acción" completos, haciendo que n8n se comporte como sus manos digitales (El sistema nervioso).

## 2. Objetivos Principales y Filosofía

1. **El Paradigma Cognitivo vs Determinista:** _"La IA decide QUÉ hacer; Python decide CÓMO hacerlo"_. Y a diferencia de asistentes aislados, Python gatilla sistemáticamente flujos de n8n vía API REST para accionar los CRMs o ecosistemas locales del cliente ("Control Inverso").
2. **Omnicanal y Proactivo (Heartbeat):** Laika no sólo reacciona a los Webhooks recibidos, sino que ejecuta "cron-jobs" internos ("Latidos") para despertar agentes que verifiquen bases de datos asíncronamente y notifiquen al N8N de forma proactiva si hay anomalías.
3. **Aislamiento Multitenant (Data Isolation):** Todo dato, request, checkpointer de la conversación o registro RAG está atado indisolublemente a un `tenant_id`. Las búsquedas en pgvector se filtrarán a nivel base de datos preventivamente (Row-Level Security) para evitar filtración trans-empresarial de Inteligencia Artificial.
4. **Resiliencia y Observabilidad (Langfuse + DLQ):** Todo "pensamiento" de agente, consumo de token e integración documentaria, será trasteado en vivo usando el dashboard Open Source **Langfuse**. Adicionalmente, si el worker de IA agota reintentos u ocurre un timeout masivo, Python avisará a n8n usando el Patrón **Dead Letter Queue (DLQ)** para avisar al humano final y no abandonarlo con un silencio infinito.
5. **Gestión de Estado Robusta y Compleja:** LangGraph guardará cada hilo asíncrono con el Checkpointer de Postgres (`AsyncPostgresSaver`). El StateGraph contendrá estructuras dedicadas como `current_intent` o `extracted_entities` separadas del propio chat histórico.
6. **Costo-Eficiencia Semántica:** Reducir gasto recurrente aplicando una **Caché Semántica estricta** (Vía pgvector en PostgreSQL) que intercepta queries redundantes antes del grafo de LangGraph.
7. **Local-First para Vectores:** Aprovechar GPU T-1000 (con modelos ultra-optimizados <1GB de VRAM como `MiniLM-L12`) para mantener la soberanía sobre los embeddings de forma nativa en español.

## 3. Requisitos de Hardware y Entorno

- **CPU:** Intel(R) Core(TM) i7-10850H CPU @ 2.70GHz
- **RAM:** 32 GB (31.8 GB usable)
- **GPU:** Quadro T-1000 (4GB VRAM) - _Exclusiva para Embeddings locales optimizados (Ej. MiniLM)_.
- **Despliegue:** 100% dockerizado (`docker-compose`).

## 4. Requerimientos Funcionales

### 4.1. Gateway Asíncrono y Encolamiento

- **API REST Asíncrona:** `FastAPI` (endpoints `/webhook`, `/status`).
- **Contrato Estricto (JSON Payload):** Todo request a FastAPI debe obligatoriamente validar bajo Pydantic: `{"tenant_id": "org_x", "thread_id": "wa_y", "channel": "whatsapp", "user_query": "...", "metadata": {}}`.
- **Worker Queues y Fire-and-Forget:** FastAPI procesa JWT, Pydantic y un `202 Aceptado` inmediato. Luego delega a Background Workers, apoyados por **Redis** (Cola de mensajes estricta y control de rotación de IA).

### 4.2. Workflows Agénticos (Los Patrones de Anthropic)

En lugar de simples agentes, estructuraremos la topología de la siguiente forma inteligente dictada por LangGraph:

1. **Router (El Portero del Costo):** Un LLM liviano inicial que desvía interacciones triviales (saludos) fuera del grafo para darles respuesta inmediata sin tocar bases de datos.
2. **Agentic RAG:** La memoria documental no se empotra masivamente. Se dota al agente de una herramienta (`RAG_Search_Tool`). El agente consulta a `pgvector`, lee la porción de texto y decide si le sirve, reformulando autónomamente su query si fue inútil.
3. **Orchestrator-Worker (Plan & Execute):** Para resolver preguntas con intenciones complejas, el Orquestador crea un Plan de Sub-tareas (Pasos). Despacha Workers que trabajan en **Paralelización**, insertando sus hallazgos en la Memoria del State de LangGraph.
4. **Evaluator-Optimizer (El Supervisor Final):** Un nodo validador revisa la respuesta propuesta, asegurándose de que cumple reglas de negocio y tono español neutro, devolviéndola a corrección ("Prompt Chaining") antes de ejecutar a Python.
5. **Prompt Injection Dinámica:** Solo inyectar el contexto estricto de la tarea que se está ejecutando para mantener el "Context Window" del LLC limpio y reducir latencia.

### 4.3. Herramientas Base (Tools) y Scripts Python

- `Semantic_Cache_Check`: Script en Python validando contra `pgvector` similitudes altas (>96%) antes de activar los Nodos IA.
- `RAG_Search_Tool`: Consulta autónoma desde la IA a la DB usando Embeddings generados por la T-1000.
- `Web_Search_Tool`: Integración con la nube (DuckDuckGo, Tavily) para conocimiento en tiempo real.
- `N8N_Workflow_Execution`: Ordena a n8n, mediante su API de integración, ejecutar automatizaciones prefabricadas completas en el mundo exterior (Ej. Cargar datos en el CRM del cliente).
- `N8N_Error_Trigger (DLQ)`: Notifica al orquestador visual (n8n) que ocurrió una falla técnica irrecuperable de Worker.

## 5. Fases de Desarrollo

### FASE 1: Fundación y Resiliencia (Enterprise Core)

- Setup `docker-compose.yml` con 7 servicios: `FastAPI`, `Celery Worker`, `Redis`, `Postgres/pgvector`, `n8n`, `Langfuse`, `Ngrok`.
- Implementar modelo determinista de variables usando `pydantic-settings`.
- Encolamiento con `Redis Workers` (Celery/RQ) garantizando tiempos de respuesta 202 hacia webhook fuente.
- Conectar LangGraph con el State Manager `langgraph-checkpoint-postgres.aio`.
- Implementación de modelo "Router" y "Evaluador-Optimizador".
- Interceptores preconfigurados structlog (NDJSON para monitoreo de fallas).
- Conexión LiteLLM (Groq, Cerebras).

### FASE 2: Advanced Agentic RAG

- Implementación del Orquestador "Plan & Execute" en flujos LangGraph.
- RAG Tool de Auto-Reformulación de Preguntas en GPU Local.
- Manejo asíncrono sub-agentes.

## 6. Filosofía del Proyecto (La Regla de Oro)

> **"La IA decide _QUÉ_ hacer; Python decide _CÓMO_ hacerlo."**

Los agentes tienen autonomía para encriptar planes de ejecución, validar información y decidir cuándo iterar, pero son nuestros robustos escudos deterministas de Python los encargados de salvaguardar el performance asíncrono y proteger al usuario de fallas insuperables de la inteligencia.
