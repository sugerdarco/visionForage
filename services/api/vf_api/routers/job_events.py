"""Bearer-authenticated SSE without retaining request database sessions."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from vf_api.dependencies import Configuration, authentication_error, bearer
from vf_api.job_stream import JobStreamResponse, parse_cursor, snapshot, stream

router = APIRouter(tags=["training jobs"])


@router.get(
    "/jobs/{job_id}/events",
    response_class=JobStreamResponse,
    status_code=200,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def job_events(
    job_id: UUID,
    request: Request,
    settings: Configuration,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    last_event_id: Annotated[str | None, Header(max_length=80)] = None,
):
    if credentials is None:
        raise authentication_error()
    after = parse_cursor(last_event_id, job_id)
    sessions = request.app.state.sessions
    initial = snapshot(
        sessions, settings, credentials.credentials, job_id, after, initial=True
    )
    slots = request.app.state.job_stream_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Too many active job streams",
            headers={"Retry-After": "2"},
        )
    return JobStreamResponse(
        stream(sessions, settings, credentials.credentials, job_id, after, initial),
        slots,
    )
