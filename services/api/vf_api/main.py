"""Application factory; importing the module opens no database connections."""

from contextlib import asynccontextmanager
import secrets
from threading import BoundedSemaphore, Lock

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from vf_api.database import make_engine, make_session_factory
from vf_api.dependencies import Database
from vf_api.routers.annotations import router as annotations_router
from vf_api.routers.imports import router as imports_router
from vf_api.routers.jobs import router as jobs_router
from vf_api.routers.models import router as models_router
from vf_api.routers.job_events import router as job_events_router
from vf_api.routers.auth import router as auth_router
from vf_api.routers.projects import router as projects_router
from vf_api.routers.uploads import router as uploads_router
from vf_api.routers.versions import router as versions_router
from vf_api.security import hash_password
from vf_api.settings import APISettings


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings)
        app.state.settings = settings
        app.state.sessions = make_session_factory(engine)
        app.state.resource_lock = Lock()
        app.state.job_stream_slots = BoundedSemaphore(32)
        try:
            app.state.dummy_password_hash = hash_password(secrets.token_hex(24))
            yield
        finally:
            for name in ("store", "queue", "datasets"):
                resource = getattr(app.state, name, None)
                if resource is not None:
                    resource.close()
            engine.dispose()

    app = FastAPI(title="VisionForge API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
    )
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(projects_router, prefix="/api/v1")
    app.include_router(uploads_router, prefix="/api/v1")
    app.include_router(annotations_router, prefix="/api/v1")
    app.include_router(imports_router, prefix="/api/v1")
    app.include_router(versions_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(job_events_router, prefix="/api/v1")

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

    async def storage_unavailable(request: Request, error: Exception):
        return JSONResponse(
            status_code=503, content={"detail": "Object storage unavailable"}
        )

    app.add_exception_handler(BotoCoreError, storage_unavailable)
    app.add_exception_handler(ClientError, storage_unavailable)

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
