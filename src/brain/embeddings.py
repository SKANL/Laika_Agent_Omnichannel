# ==========================================
# EMBEDDINGS: SINGLETON GPU LOCAL (T-1000)
# ==========================================
# El modelo se carga UNA sola vez en memoria al primer acceso.
# En desarrollo: detecta automáticamente si hay GPU disponible (CUDA/CPU).
# En producción: forzar device="cuda:0" vía variable de entorno.
#
# Modelo elegido: paraphrase-multilingual-MiniLM-L12-v2
#   - 384 dimensiones (coincide con Vector(384) en db.py)
#   - <500MB VRAM — cabe en la Quadro T-1000 con margen
#   - Soporta español nativo (multilingüe)
#   - Licencia: Apache 2.0

import os
from typing import Optional
import structlog

logger = structlog.get_logger("laika_embeddings")

_model_instance = None
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def get_embedding_model():
    """
    Devuelve el singleton del modelo de embeddings.
    Carga lazy: el modelo solo se descarga/mueve a GPU en la primera llamada.
    Thread-safe para uso en Celery workers (cada proceso tiene su propio singleton).
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        # Auto-detección de device: CUDA > MPS (Apple) > CPU
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("embedding_model_loading", model=_MODEL_NAME, device="cuda",
                        gpu=torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("embedding_model_loading", model=_MODEL_NAME, device="mps")
        else:
            device = "cpu"
            logger.warning("embedding_model_loading_cpu", model=_MODEL_NAME, device="cpu",
                           note="No GPU found. Using CPU — embeddings will be slower.")

        _model_instance = SentenceTransformer(_MODEL_NAME, device=device)
        logger.info("embedding_model_ready", model=_MODEL_NAME, device=device,
                    dimensions=_model_instance.get_sentence_embedding_dimension())
        return _model_instance

    except ImportError:
        logger.error("sentence_transformers_not_installed",
                     hint="pip install sentence-transformers torch")
        raise


def encode_text(text: str) -> list[float]:
    """
    Convierte texto a vector de embeddings normalizado.
    Usado por rag_tool.py, cache.py y el endpoint de ingesta de documentos.

    Returns:
        list[float] de longitud 384 (paraphrase-multilingual-MiniLM-L12-v2)
    """
    model = get_embedding_model()
    # normalize_embeddings=True garantiza similitud coseno correcta con pgvector
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
