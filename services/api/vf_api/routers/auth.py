"""JSON account endpoints. Blocking database and bcrypt work runs in sync routes."""

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vf_api.dependencies import (
    Configuration,
    CurrentUser,
    Database,
    authentication_error,
    token_user,
)
from vf_api.models import User
from vf_api.schemas import (
    AccessTokenResponse,
    Credentials,
    LoginResponse,
    RefreshRequest,
    UserResponse,
)
from vf_api.security import hash_password, issue_token, verify_password

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=201)
def register(body: Credentials, session: Database) -> User:
    user = User(
        email=body.email,
        password_hash=hash_password(body.password.get_secret_value()),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        if (
            getattr(getattr(error.orig, "diag", None), "constraint_name", None)
            == "uq_users_email"
        ):
            raise HTTPException(
                status_code=409, detail="Email is already registered"
            ) from None
        raise
    return user


@router.post("/login", response_model=LoginResponse)
def login(
    body: Credentials, request: Request, session: Database, settings: Configuration
) -> LoginResponse:
    user = session.scalar(select(User).where(User.email == body.email))
    # Unknown accounts still perform a bcrypt check before the same 401 response.
    stored_hash = user.password_hash if user else request.app.state.dummy_password_hash
    valid = verify_password(body.password.get_secret_value(), stored_hash)
    if not valid or user is None:
        raise authentication_error()
    access, access_seconds = issue_token(user.id, "access", settings)
    refresh, refresh_seconds = issue_token(user.id, "refresh", settings)
    return LoginResponse(
        access_token=access,
        expires_in=access_seconds,
        refresh_token=refresh,
        refresh_expires_in=refresh_seconds,
    )


@router.post("/refresh", response_model=AccessTokenResponse)
def refresh(
    body: RefreshRequest, session: Database, settings: Configuration
) -> AccessTokenResponse:
    user = token_user(
        body.refresh_token.get_secret_value(), "refresh", session, settings
    )
    access, seconds = issue_token(user.id, "access", settings)
    # The original refresh token keeps its fixed expiry; refreshing cannot extend it.
    return AccessTokenResponse(access_token=access, expires_in=seconds)


@router.get("/me", response_model=UserResponse)
def me(user: CurrentUser) -> User:
    return user
