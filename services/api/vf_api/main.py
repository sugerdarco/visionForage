"""Application factory; importing the module opens no database connections."""

from contextlib import asynccontextmanager
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from vf_api.database import make_engine, make_session_factory
from vf_api.dependencies import Database
from vf_api.routers.auth import router as auth_router
from vf_api.routers.projects import router as projects_router
from vf_api.security import hash_password
from vf_api.settings import APISettings


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings)
        app.state.settings = settings
        app.state.sessions = make_session_factory(engine)
        try:
            app.state.dummy_password_hash = hash_password(secrets.token_hex(24))
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="VisionForge API", version="0.1.0", lifespan=lifespan)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(projects_router, prefix="/api/v1")

    @app.middleware("http")
    async def private_api_responses(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # FastAPI's default includes rejected input, which may contain credentials.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: item[key] for key in ("loc", "msg", "type")}
                    for item in error.errors()
                ]
            },
        )

    @app.exception_handler(OperationalError)
    async def database_unavailable(request: Request, error: OperationalError):
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

    @app.get("/healthz", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    def ready(session: Database) -> dict[str, str]:
        # Verify both connectivity and the table needed by authentication.
        try:
            session.execute(text("SELECT id FROM users LIMIT 0"))
        except SQLAlchemyError:
            raise HTTPException(
                status_code=503, detail="Database unavailable"
            ) from None
        return {"status": "ready"}

    return app
