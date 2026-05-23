from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

_SECRET = os.getenv("JWT_SECRET")
if not _SECRET:
    raise RuntimeError("JWT_SECRET is not set in environment")
_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
_EXPIRY_DAYS = int(os.getenv("JWT_EXPIRY_DAYS", "30"))

_bearer = HTTPBearer()


class TokenClaims(BaseModel):
    user_id: int
    email: str
    username: str
    is_admin: bool = False


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: int, email: str, username: str, is_admin: bool = False) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "username": username,
        "is_admin": is_admin,
        "exp": datetime.now(timezone.utc) + timedelta(days=_EXPIRY_DAYS),
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def get_claims(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> TokenClaims:
    try:
        payload = jwt.decode(credentials.credentials, _SECRET, algorithms=[_ALGORITHM])
        return TokenClaims(
            user_id=int(payload["sub"]),
            email=payload["email"],
            username=payload["username"],
            is_admin=payload.get("is_admin", False),
        )
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(claims: TokenClaims = Depends(get_claims)) -> TokenClaims:
    if not claims.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return claims
