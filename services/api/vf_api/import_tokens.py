"""Archive grants have their own audience, separate from images and access tokens."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt

from vf_api.import_schemas import ImportClaim
from vf_api.security import ISSUER
from vf_api.settings import APISettings
from vf_api.storage import UPLOAD_SECONDS


def grant_import(
    file: ImportClaim, project_id: UUID, user_id: UUID, settings: APISettings
) -> tuple[str, str]:
    identifier = str(uuid4())
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": "visionforge-import",
            "type": "import",
            "sub": str(user_id),
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
    return token, f"uploads/imports/{project_id}/{identifier}"


def read_import(
    token: str, project_id: UUID, user_id: UUID, settings: APISettings
) -> tuple[ImportClaim, str]:
    claims = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=["HS256"],
        issuer=ISSUER,
        audience="visionforge-import",
        options={
            "require": [
                "iss",
                "aud",
                "type",
                "sub",
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
        claims["type"] != "import"
        or claims["sub"] != str(user_id)
        or claims["project_id"] != str(project_id)
    ):
        raise jwt.InvalidTokenError("Import scope does not match")
    try:
        identifier = UUID(claims["jti"])
        file = ImportClaim.model_validate(claims["file"])
    except (ValueError, TypeError, AttributeError):
        raise jwt.InvalidTokenError("Invalid import claims") from None
    return file, f"uploads/imports/{project_id}/{identifier}"
