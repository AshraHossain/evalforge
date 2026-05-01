"""
api/auth.py — JWT Authentication

WHY THIS FILE EXISTS:
    All EvalForge API endpoints are protected.  A submitted eval job might
    trigger expensive LLM calls; without auth, anyone could spam your server.

    FLOW:
    1. Client calls POST /auth/token with username+password
    2. auth.py verifies credentials, returns a JWT access token
    3. Client includes `Authorization: Bearer <token>` on every request
    4. `get_current_user` dependency decodes the token and injects the user

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/auth.py ───────────────────────────────────────────────────────────┐
    │  Used by:     api/main.py (router registration)                        │
    │               api/routers/*.py (Depends(get_current_user))             │
    │  Uses:        config.py (JWT_SECRET_KEY, JWT_ALGORITHM)                │
    └─────────────────────────────────────────────────────────────────────────┘

    NOTE: This implementation uses a simple in-memory user store for Phase 1.
    In production, replace with a proper user table in PostgreSQL.
"""

import base64
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Suppress passlib's noisy bcrypt version-detection warning (harmless with bcrypt<4)
logging.getLogger("passlib").setLevel(logging.ERROR)

from config import settings

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prepare(password: str) -> str:
    """SHA-256 pre-hash so bcrypt never sees >72 bytes (its hard limit).

    bcrypt silently truncates inputs longer than 72 bytes, which makes two
    different long passwords hash identically — a security hole.  By
    pre-hashing with SHA-256 we normalise every password to exactly 44
    base64 chars before bcrypt sees it.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Simple in-memory user store (replace with DB in production) ────────────────
# Password hash for "evalforge123" — change before deploying
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash(_prepare("evalforge123")),
        "email": "admin@evalforge.local",
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_prepare(plain_password), hashed_password)


def get_user(username: str) -> Optional[dict]:
    return FAKE_USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    FastAPI dependency.  Validates the JWT and returns the current user.

    Usage in any router:
        @router.get("/protected")
        async def protected(user: User = Depends(get_current_user)):
            ...
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    return User(username=user["username"], email=user.get("email"))


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Exchange username+password for a JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["username"]})
    return Token(access_token=access_token, token_type="bearer")
