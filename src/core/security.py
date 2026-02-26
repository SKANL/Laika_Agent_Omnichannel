# ==========================================
# SEGURIDAD: VERIFICACIÓN JWT (python-jose)
# ==========================================
# Dependency inyectable en FastAPI para proteger endpoints.
# N8N y clientes externos deben incluir el header:
#   Authorization: Bearer <token>
#
# El token es firmado con JWT_SECRET_KEY del .env.
# Para generar un token de prueba:
#   python -c "from jose import jwt; print(jwt.encode({'sub':'test','tenant_id':'demo'}, 'TU_KEY', algorithm='HS256'))"

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from src.core.config import settings
import structlog

logger = structlog.get_logger("laika_security")

_bearer_scheme = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme)) -> dict:
    """
    FastAPI Dependency: valida el JWT Bearer token del request entrante.

    Uso en endpoints:
        @router.post("/endpoint")
        async def my_endpoint(payload: ..., claims: dict = Depends(verify_token)):
            tenant_from_token = claims.get("tenant_id")

    Raises:
        HTTPException 401 si el token falta, está expirado o la firma es inválida.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header requerido. Incluye: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    secret = settings.JWT_SECRET_KEY.get_secret_value()

    try:
        claims = jwt.decode(token, secret, algorithms=[ALGORITHM])
        logger.debug("jwt_verified", sub=claims.get("sub"))
        return claims
    except JWTError as exc:
        logger.warning("jwt_verification_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token inválido o expirado: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_dev_token(tenant_id: str, sub: str = "dev") -> str:
    """
    Genera un token JWT de desarrollo/prueba.
    NO usar en producción sin agregar 'exp' (expiración).
    """
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    payload = {"sub": sub, "tenant_id": tenant_id, "scope": "webhook:write"}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)
