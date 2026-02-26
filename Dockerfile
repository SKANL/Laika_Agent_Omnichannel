# ==========================================
# LAIKA V2: API GATEWAY MULTI-STAGE DOCKERFILE
# ==========================================

# 1. ETAPA DE CONSTRUCCIÓN (BUILDER)
# Usamos una imagen slim para descargar y copilar lo necesario
FROM python:3.12-slim as builder

WORKDIR /app

# Variables para que Python no escriba .pyc y flushie rápido
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para compilar asyncpg o drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear y activar entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar requirements y compilar en el venv aislado
COPY requirements.txt .
# Instalar torch CPU ANTES del resto para evitar que PyPI resuelva la variante CUDA
# torch CPU (linux/amd64) pesa ~200 MB vs ~4 GB de la variante CUDA por defecto
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# 2. ETAPA FINAL (RUNNER)
# Imagen más ligera y limpia para correr en producción
FROM python:3.12-slim as runner

WORKDIR /app

# Ocultar advertencias y mejorar flushes de Logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Dependencia vital para que psycopg corra
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Traer el entorno virtual compilado de la etapa anterior
COPY --from=builder /opt/venv /opt/venv

# Copiar código fuente
COPY ./src /app/src

# Puerto FastAPI por defecto
EXPOSE 8000

# Ejecutar el orquestador
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
