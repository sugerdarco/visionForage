"""Request-scoped sessions and the ownership identity for protected routes."""

from collections.abc import Iterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.models import Project, User
from vf_api.security import TokenKind, read_subject
from vf_api.settings import APISettings

bearer = HTTPBearer(auto_error=False)


def get_settings(request: Request) -> APISettings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


Database = Annotated[Session, Depends(get_session)]
Configuration = Annotated[APISettings, Depends(get_settings)]


def authentication_error() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def token_user(
    token: str, kind: TokenKind, session: Session, settings: APISettings
) -> User:
    try:
        user_id = read_subject(token, kind, settings)
    except jwt.InvalidTokenError:
        raise authentication_error() from None
    user = session.get(User, user_id)
    if user is None:
        raise authentication_error()
    return user


def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: Database,
    settings: Configuration,
) -> User:
    if credentials is None:
        raise authentication_error()
    return token_user(credentials.credentials, "access", session, settings)


CurrentUser = Annotated[User, Depends(current_user)]


def owned_project(
    project_id: UUID, session: Session, user: User, *, lock: bool = False
) -> Project:
    statement = select(Project).where(
        Project.id == project_id, Project.owner_id == user.id
    )
    if lock:
        statement = statement.with_for_update()
    project = session.scalar(statement)
    if project is None:
        # A foreign ID and a missing ID have the same public response.
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def get_project(project_id: UUID, session: Database, user: CurrentUser) -> Project:
    return owned_project(project_id, session, user)


def get_locked_project(
    project_id: UUID, session: Database, user: CurrentUser
) -> Project:
    return owned_project(project_id, session, user, lock=True)


OwnedProject = Annotated[Project, Depends(get_project)]
LockedProject = Annotated[Project, Depends(get_locked_project)]
