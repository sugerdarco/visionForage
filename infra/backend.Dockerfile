FROM docker:29.9.1-cli@sha256:6fae2214c3121a28d5aa639d09032f61310b7c9c75ffbad1025f261a1b621800 AS docker_cli
FROM python:3.11.16-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    PYTHONPATH=/app/services/api:/app/services/worker:/app/packages
WORKDIR /app
COPY requirements/api.txt /tmp/api.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --timeout 120 --retries 5 --disable-pip-version-check -r /tmp/api.txt \
    && pip check && rm /tmp/api.txt
COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
COPY services/api/vf_api /app/services/api/vf_api
COPY services/api/alembic /app/services/api/alembic
COPY services/api/alembic.ini /app/services/api/alembic.ini
COPY services/worker/vf_worker /app/services/worker/vf_worker
COPY packages/vf_ml /app/packages/vf_ml
USER 1000:1000
CMD ["python", "-m", "uvicorn", "vf_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
