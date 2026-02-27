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

from datetime import datetime, timedelta

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


def generate_dev_token(
    tenant_id: str,
    sub: str = "dev",
    expire_days: int | None = None,
) -> str:
    """
    Genera un token JWT firmado con expiración.
    Por defecto expira en JWT_EXPIRE_DAYS días (configurado en settings).

    Args:
        tenant_id: ID del tenant propietario del token.
        sub:       Sujeto del token (ej. email o user_id).
        expire_days: Días hasta la expiración. None = usar settings.JWT_EXPIRE_DAYS.
    """
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    days = expire_days if expire_days is not None else settings.JWT_EXPIRE_DAYS
    payload = {
        "sub": sub,
        "tenant_id": tenant_id,
        "scope": "webhook:write",
        "exp": datetime.utcnow() + timedelta(days=days),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)
