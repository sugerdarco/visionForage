"""Password hashing and narrowly scoped JWT verification."""

from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

import bcrypt
import jwt

from vf_api.settings import APISettings

TokenKind = Literal["access", "refresh"]
ISSUER = "visionforge"
AUDIENCE = "visionforge-api"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode(
        "ascii"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, UnicodeError):
        return False


def issue_token(
    user_id: UUID, kind: TokenKind, settings: APISettings
) -> tuple[str, int]:
    lifetime = (
        timedelta(minutes=settings.access_token_minutes)
        if kind == "access"
        else timedelta(days=settings.refresh_token_days)
    )
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": str(user_id),
            "type": kind,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now,
            "nbf": now,
            "exp": now + lifetime,
            "jti": str(uuid4()),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    return token, int(lifetime.total_seconds())


def read_subject(token: str, kind: TokenKind, settings: APISettings) -> UUID:
    if len(token) > 4096:
        raise jwt.InvalidTokenError("Token is too long")
    claims = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        issuer=ISSUER,
        audience=AUDIENCE,
        options={"require": ["sub", "type", "iss", "aud", "iat", "nbf", "exp", "jti"]},
    )
    if claims["type"] != kind:
        raise jwt.InvalidTokenError("Wrong token type")
    try:
        return UUID(claims["sub"])
    except (ValueError, TypeError, AttributeError) as error:
        raise jwt.InvalidTokenError("Invalid subject") from error
