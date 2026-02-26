import logging
import sys
import structlog

# ==========================================
# LAIKA V2: LOGGING ESTRUCTURADO Y JSON
# ==========================================
# Permite tracear cada nodo asíncrono en Kubernetes o Docker
# mediante diccionarios en lugar de texto plano caótico.

def setup_logging(json_logs: bool = True):
    """
    Configura Structlog para que la API entera emita NDJSON
    o consola coloreada dependiendo del entorno.
    """
    # Nivel base del logger nativo
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Filtros para agregar marca de tiempo, nivel, etc.
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Renderizador final: JSON crudo vs Consola bonita
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Redireccionar el logger estándar por la tubería de Structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

# Instanciamos el logger global que todos usarán
logger = structlog.get_logger("laika_gateway")
