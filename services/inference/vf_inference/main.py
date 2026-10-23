"""Bounded, API-key-protected inference for one CPU model bundle."""

from contextlib import asynccontextmanager
import asyncio
import hashlib
import secrets
from threading import BoundedSemaphore
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from vf_inference.runtime import ModelRuntime
from vf_inference.settings import Settings

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Box(BaseModel):
    label: str
    confidence: float
    x: float
    y: float
    w: float
    h: float


class Prediction(BaseModel):
    request_id: UUID
    model_id: UUID
    model_version: int
    latency_ms: float


class ClassificationPrediction(Prediction):
    label: str
    confidence: float
    scores: dict[str, float]


class DetectionPrediction(Prediction):
    boxes: list[Box]


async def bounded_stream(request):
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Image upload too large")
        yield chunk


async def uploaded_image(request):
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip()
        != "multipart/form-data"
    ):
        raise HTTPException(
            status_code=415, detail="Use multipart/form-data with one image file"
        )
    parser = MultiPartParser(
        request.headers, bounded_stream(request), max_files=1, max_fields=0
    )
    try:
        form = await parser.parse()
    except MultiPartException:
        raise HTTPException(status_code=422, detail="Invalid image upload") from None
    try:
        entries = form.multi_items()
        if (
            len(entries) != 1
            or entries[0][0] != "image"
            or not isinstance(entries[0][1], UploadFile)
        ):
            raise HTTPException(
                status_code=422, detail="Provide exactly one image file"
            )
        upload = entries[0][1]
        if upload.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=415, detail="Only JPEG and PNG images are supported"
            )
        data = await upload.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image upload too large")
        if not data:
            raise HTTPException(status_code=422, detail="Empty image")
        return data, upload.content_type
    finally:
        await form.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.runtime = ModelRuntime(
                settings.bundle_dir,
                settings.model_sha256,
                settings.metadata_sha256,
                cpu_threads=settings.cpu_threads,
            )
        except Exception:
            raise RuntimeError(
                "Cannot initialize the verified inference bundle"
            ) from None
        app.state.slot = BoundedSemaphore(1)
        try:
            yield
        finally:
            del app.state.runtime

    app = FastAPI(title="VisionForge Inference", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz")
    def ready(request: Request):
        if not hasattr(request.app.state, "runtime"):
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"status": "ready"}

    @app.post(
        "/predict/{slug}",
        response_model=ClassificationPrediction | DetectionPrediction,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["image"],
                            "properties": {
                                "image": {"type": "string", "format": "binary"}
                            },
                        }
                    }
                },
            }
        },
    )
    async def predict(
        slug: str,
        request: Request,
        api_key: Annotated[str | None, Security(key_header)],
    ):
        if (
            not api_key
            or not 32 <= len(api_key) <= 512
            or not secrets.compare_digest(
                hashlib.sha256(api_key.encode()).hexdigest(),
                settings.api_key_sha256.get_secret_value(),
            )
        ):
            raise HTTPException(status_code=401, detail="Invalid API key")
        if slug != settings.slug:
            raise HTTPException(status_code=404, detail="Deployment not found")
        slot = request.app.state.slot
        if not slot.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Inference busy; retry shortly",
                headers={"Retry-After": "1"},
            )
        try:
            try:
                async with asyncio.timeout(30):
                    data, mime = await uploaded_image(request)
            except TimeoutError:
                raise HTTPException(
                    status_code=408, detail="Image upload timed out"
                ) from None
            start = perf_counter()
            try:
                result = await run_in_threadpool(
                    request.app.state.runtime.predict, data, mime
                )
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="Invalid or unsupported image"
                ) from None
            except Exception:
                raise HTTPException(
                    status_code=503, detail="Inference unavailable"
                ) from None
            return {
                "request_id": uuid4(),
                "model_id": settings.model_id,
                "model_version": settings.model_version,
                "latency_ms": (perf_counter() - start) * 1000,
                **result,
            }
        finally:
            slot.release()

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    return app
