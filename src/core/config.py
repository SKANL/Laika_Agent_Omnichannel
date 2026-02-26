from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, IPvAnyAddress, HttpUrl, Field
from typing import Optional

# ==========================================
# GESTOR CENTRAL DE CONFIGURACIÓN (SINGLETON)
# ==========================================
# Pydantic Settings carga nuestro archivo .env de forma
# fuertemente tipada. Evitamos el uso anticuado de os.getenv.

class Settings(BaseSettings):
    # El archivo origen para buscar variables
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 1. Postgres & RAG
    POSTGRES_USER: str = "laika_admin"
    POSTGRES_PASSWORD: str = "laika_secure_pass_2026"
    POSTGRES_DB: str = "laika_db"
    
    # 2. Infra y Tokens
    JWT_SECRET_KEY: SecretStr = Field(..., description="JWT firma")
    # REDIS_URL debe incluir la contraseña cuando Redis usa --requirepass.
    # Formato: redis://:password@host:port/db  (los dos puntos antes de password son obligatorios)
    # El docker-compose inyecta esta URL con la contraseña correcta en cada contenedor.
    REDIS_URL: str = Field(default="redis://:laika_redis_secure_2026@localhost:6379/0", description="Cola Worker Celery (incluye auth)")
    TIMEZONE: str = Field(default="America/Mexico_City", description="Timezone para Celery Beat")

    # 3. LLMs (Tiers)
    GROQ_API_KEY: SecretStr = Field(..., description="API Velocistas")
    CEREBRAS_API_KEY: SecretStr = Field(..., description="API Heavy Lifters")

    # 4. Observabilidad (Langfuse v3)
    # IMPORTANTE: En Langfuse v3, la variable se llama LANGFUSE_BASE_URL, NO LANGFUSE_HOST
    LANGFUSE_BASE_URL: str = Field(default="http://localhost:3000")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_PUBLIC_KEY: str = Field(default="")

    # 5. Reverse Control (n8n API) & External Access
    N8N_WEBHOOK_URL: HttpUrl = Field(default="http://localhost:5678/")
    N8N_API_KEY: SecretStr = Field(default="", description="Token para autenticarse contra el API/Webhook de n8n")

    @property
    def sync_database_url(self) -> str:
        """Driver síncrono para LangGraph PostgresSaver base (si llegara a ocuparse)"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@postgres:5432/{self.POSTGRES_DB}"
    
    @property
    def async_database_url(self) -> str:
        """Driver asíncrono para SQLAlchemy (usa asyncpg como driver)"""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@postgres:5432/{self.POSTGRES_DB}"

    @property
    def psycopg_database_url(self) -> str:
        """URL estándar para psycopg/AsyncConnectionPool de LangGraph.
        psycopg NO acepta el prefijo '+asyncpg' (eso es formato SQLAlchemy)."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@postgres:5432/{self.POSTGRES_DB}"

# Al inicializar este objeto al momento de importar este archivo,
# validará mágicamente que nuestro ".env" esté en orden y sin roturas.
settings = Settings()
