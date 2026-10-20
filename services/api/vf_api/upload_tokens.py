"""Signed grants bind a temporary object to one owner, project and file digest."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt
from pydantic import ValidationError

from vf_api.security import ISSUER
from vf_api.settings import APISettings
from vf_api.storage import UPLOAD_SECONDS
from vf_api.upload_schemas import FileClaim


def grant_upload(
    file: FileClaim, project_id: UUID, user_id: UUID, settings: APISettings
) -> tuple[str, str]:
    identifier = str(uuid4())
    now = datetime.now(timezone.utc)
    key = f"uploads/{project_id}/{identifier}"
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": "visionforge-upload",
            "sub": str(user_id),
            "type": "upload",
            "project_id": str(project_id),
            "jti": identifier,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(seconds=UPLOAD_SECONDS),
            "file": file.model_dump(),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    return token, key


def read_upload(
    token: str, project_id: UUID, user_id: UUID, settings: APISettings
) -> tuple[FileClaim, str]:
    claims = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        issuer=ISSUER,
        audience="visionforge-upload",
        options={
            "require": [
                "iss",
                "aud",
                "sub",
                "type",
                "project_id",
                "jti",
                "iat",
                "nbf",
                "exp",
                "file",
            ]
        },
    )
    if (
        claims["type"] != "upload"
        or claims["sub"] != str(user_id)
        or claims["project_id"] != str(project_id)
    ):
        raise jwt.InvalidTokenError("Upload scope does not match")
    try:
        identifier = UUID(claims["jti"])
        file = FileClaim.model_validate(claims["file"])
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise jwt.InvalidTokenError("Invalid upload claims") from None
    return file, f"uploads/{project_id}/{identifier}"
