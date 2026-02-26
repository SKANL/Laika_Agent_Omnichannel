# ==========================================
# INGESTA DE DOCUMENTOS (Pipeline RAG)
# ==========================================
# Endpoint para subir texto a la base de conocimientos documental por tenant.
# El pipeline:
#   1. Recibe texto + tenant_id + metadatos (JWT autenticado)
#   2. Divide el texto en chunks si es largo (simple split)
#   3. Genera embeddings con el modelo local (GPU T-1000)
#   4. Almacena en rag_documents con el tenant_id como RLS

from fastapi import APIRouter, status, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from structlog import get_logger

from src.core.db import AsyncSessionLocal, RAGDocument
from src.core.security import verify_token

logger = get_logger("laika_documents")
router = APIRouter(prefix="/v1/documents", tags=["Document Ingestion"])

# Tamano maximo de chunk en caracteres (~300 palabras)
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 100


# --- Schemas ---
class IngestRequest(BaseModel):
    tenant_id: str = Field(..., min_length=3, description="ID del tenant dueno de este conocimiento")
    content: str = Field(..., min_length=20, description="Texto del documento a indexar")
    source: Optional[str] = Field(default="manual_upload", description="Origen del documento (ej. manual_v2.pdf)")
    tags: Optional[List[str]] = Field(default_factory=list, description="Etiquetas para organizar el conocimiento")


class IngestResponse(BaseModel):
    status: str
    tenant_id: str
    chunks_indexed: int
    message: str


# --- Chunking simple (Fase 1: sin LangChain TextSplitter) ---
def _split_into_chunks(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """
    Divide el texto en chunks con overlap para evitar cortar contextos importantes.
    En Fase 2: reemplazar con RecursiveCharacterTextSplitter de LangChain.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Intentar cortar en un punto o salto de linea para no partir palabras
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut == -1:
                cut = text.rfind(". ", start, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    req: IngestRequest,
    claims: dict = Depends(verify_token),
):
    """
    Indexa un documento en la base de conocimientos del tenant.

    Flujo:
    1. Autentica via JWT (verify_token).
    2. Divide el texto en chunks solapados.
    3. Genera embeddings con SentenceTransformers (GPU T-1000 en dev).
    4. Inserta en rag_documents con tenant_id para RLS.

    Ejemplo de uso:
        curl -X POST /v1/documents/ingest \\
          -H "Authorization: Bearer <token>" \\
          -H "Content-Type: application/json" \\
          -d '{"tenant_id":"org_x","content":"Manual de producto...","source":"manual_v1.pdf"}'
    """
    from src.brain.embeddings import encode_text

    logger.info("document_ingest_start", tenant=req.tenant_id,
                source=req.source, content_len=len(req.content))

    chunks = _split_into_chunks(req.content)
    logger.info("document_chunked", tenant=req.tenant_id, chunks=len(chunks))

    metadata = {"source": req.source, "tags": req.tags or []}

    async with AsyncSessionLocal() as session:
        try:
            for i, chunk in enumerate(chunks):
                embedding = encode_text(chunk)
                doc = RAGDocument(
                    tenant_id=req.tenant_id,
                    content=chunk,
                    embedding=embedding,
                    metadata_json={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                )
                session.add(doc)

            await session.commit()
            logger.info("document_ingest_success", tenant=req.tenant_id, chunks=len(chunks))

        except Exception as e:
            await session.rollback()
            logger.error("document_ingest_failed", error=str(e), tenant=req.tenant_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al indexar documento: {str(e)}",
            )

    return IngestResponse(
        status="indexed",
        tenant_id=req.tenant_id,
        chunks_indexed=len(chunks),
        message=f"Documento indexado correctamente en {len(chunks)} chunk(s).",
    )


@router.delete("/tenant/{tenant_id}", status_code=status.HTTP_200_OK)
async def delete_tenant_documents(
    tenant_id: str,
    claims: dict = Depends(verify_token),
):
    """
    Elimina TODOS los documentos RAG de un tenant.
    Util para re-indexar desde cero.
    """
    from sqlalchemy import delete

    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(
                delete(RAGDocument).where(RAGDocument.tenant_id == tenant_id)
            )
            await session.commit()
            deleted = result.rowcount
            logger.info("tenant_docs_deleted", tenant=tenant_id, count=deleted)
            return {"status": "deleted", "tenant_id": tenant_id, "documents_removed": deleted}
        except Exception as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
