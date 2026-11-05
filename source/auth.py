import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from source.database import get_db
from source.models import User


load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480")
)

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from the environment.")
if len(JWT_SECRET_KEY) < 32:
    raise RuntimeError("JWT_SECRET_KEY must contain at least 32 characters.")

if not 5 <= ACCESS_TOKEN_EXPIRE_MINUTES <= 1440:
    raise RuntimeError(
        "ACCESS_TOKEN_EXPIRE_MINUTES must be between 5 and 1440."
    )

password_hash = PasswordHash.recommended()
_dummy_password_hash = password_hash.hash(secrets.token_urlsafe(32))

# FastAPI uses this OAuth2 scheme to expose a real login form in Swagger UI.
bearer_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
    auto_error=False,
)


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def valid_email(email: str) -> bool:
    normalized = normalize_email(email)
    return bool(
        len(normalized) <= 320
        and re.fullmatch(
            r"[^@\s]+@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
            normalized,
        )
    )


def hash_password(password: str) -> str:
    if not 12 <= len(password) <= 128:
        raise ValueError("Password must be between 12 and 128 characters.")
    return password_hash.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return password_hash.verify(password, stored_hash)
    except Exception:
        return False


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": expires_at,
        "type": "access",
        "iss": "aerointel",
        "aud": "aerointel-api",
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def get_current_user(
    token: str | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise unauthorized

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            issuer="aerointel",
            audience="aerointel-api",
        )
        user_id = payload.get("sub")
        if payload.get("type") != "access" or not isinstance(user_id, str):
            raise unauthorized
    except jwt.InvalidTokenError:
        raise unauthorized

    user = db.scalar(select(User).where(User.id == user_id))
    if user is None or not user.is_active:
        raise unauthorized

    return user


def verify_login_password(password: str, stored_hash: str | None) -> bool:
    """Always perform a password hash check to reduce email enumeration."""
    return verify_password(
        password,
        stored_hash if stored_hash is not None else _dummy_password_hash,
    )
