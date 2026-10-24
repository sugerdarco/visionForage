# Running the complete local backend

After the one-time storage tool build in [LOCAL.md](LOCAL.md), run:

```bash
make env
make up
make status
```

`make up` now builds the backend and inference images and starts PostgreSQL, Redis,
MinIO, Traefik, the API, the CPU worker and one Beat scheduler. One-shot services
create private buckets, configure temporary-upload expiry and apply Alembic
migrations before the API/workers start. Repeating startup preserves existing data.
No host API environment is required for this path; Docker Compose, Make and Python 3
are required. Initial builds download pinned images and Python packages.

The API is at `http://127.0.0.1:8000`, with interactive documentation at `/docs`
and dependency readiness at `/readyz`. Browser uploads use
`http://127.0.0.1:9000`; inference uses
`http://127.0.0.1:8080/predict/{slug}`. All published ports bind loopback.
Set `API_PORT`, `MINIO_API_PORT` and `TRAEFIK_PORT` in private `.env` to change them;
keep `DEPLOYMENT_GATEWAY_URL` aligned with the gateway's host port.

## How the processes connect

The API writes metadata to PostgreSQL and schedules CPU work through Redis.
The CPU worker generates thumbnails, dispatches GPU jobs, recovers expired jobs,
collects prediction logs and cleans deleted projects. Beat submits recurring tasks.
The remote GPU worker continues to use its user-owned Python environment and
private forwarded connections; it needs no Docker or sudo. Update its checkout
and restart the worker when deploying worker changes, as described in [GPU_WORKER.md](GPU_WORKER.md).

Containers use `postgres`, `redis` and `minio` service names internally. Presigned
URLs use `S3_PUBLIC_ENDPOINT_URL`, defaulting to the host MinIO port, so the browser
can reach them. The API probes inference routes through `http://traefik:8000`,
while returned prediction URLs use the public gateway origin.

The backend image includes the API, CPU worker and Docker CLI, using pinned API
requirements without PyTorch. One image keeps migrations and services on the same
code revision. A separate inference image supplies ONNX Runtime and serving code;
new deployments resolve and use the locally built image. Heavy training stays on
the GPU VM. These choices keep local CPU services independent of CUDA compatibility.

## Local paths and identity

`make backend-env` prepares private directories and records `ARTIFACTS_DIR`,
`LOCAL_UID` and `LOCAL_GID` in the ignored `.env`, preserving credentials. The API
and CPU worker use the host user's identity. `DOCKER_GID` defaults to 0 for the
local Docker Desktop socket; on Docker Engine set it to the socket's actual group
if needed. This grants socket access without running the application as root.
The socket gives these trusted control services authority over the local daemon.

Artifact subdirectories are mounted at the same absolute paths inside and outside
the backend containers. This matters because Docker resolves inference bind
mounts on the daemon host, as described in the [Docker bind-mount documentation](https://docs.docker.com/engine/storage/bind-mounts/).
Native `DEPLOYMENT_WORK_DIR` and `PREDICTION_LOG_DIR` settings must resolve to the
`deployments` and `predictions` subdirectories of `ARTIFACTS_DIR`. Remote Docker
contexts are unsupported. Keep paths and deployment namespace stable while models
are deployed. Backend containers have a read-only root, private writable mounts,
a temporary directory, an init process to reap children, dropped capabilities and
bounded rotating logs. Beat keeps its disposable schedule file in container tmpfs:
host-mounted database files blocked scheduler shutdown on the tested Docker Desktop
filesystem. A restart resets periodic timing; durable jobs and cleanup requests
remain in PostgreSQL and the next scheduled scan recovers them.

## Operation

Stop native API/CPU worker/Beat processes before using their Compose replacements;
run only one Beat scheduler for this installation. `make api`, `make worker` and
`make beat` remain available for native development with the API virtual environment.
Do not start native `make api` alongside the container on the same port.

```bash
# Rebuild and apply source changes:
make up
# Inspect service logs; keep raw logs private:
docker compose --env-file .env -f infra/compose.yml logs --tail 50 api worker-cpu beat
# Stop application services while retaining volumes:
make down
```

Migrations and storage setup use Compose completion dependencies; API and workers
wait for them to succeed. See [Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/).
Model containers are managed dynamically; stop them through the API before
shutting down the stack. `make down` preserves database/storage volumes and local
artifacts, and does not delete those model containers.

## Completion boundary

The backend includes accounts, projects, uploads/imports, annotation, dataset
versions, GPU training and recovery, live events, model reports/downloads/promotion,
Docker deployment, prediction logging/monitoring, [human review](REVIEWS.md) and
[populated-project cleanup](PROJECT_DELETION.md). Frontend implementation, the
active-learning comparison notebook and final browser workflow verification remain.

## Verification

On 2026-10-24, the packaged API, CPU worker and Beat completed a real HTTP flow:
six images uploaded, labeled and thumbnailed; a synthetic ONNX serving fixture
deployed through Traefik; one prediction automatically collected and reviewed;
a second frozen version containing seven images; and automatic project deletion
removing its model container, local files, stored objects and metadata. The first
frozen version still contained six images before deletion. Repeating acceptance
returned the same receipt. This checks integration, not model quality or new GPU
training; earlier real classifier/detector and GPU checks remain documented.

The regression suites passed 98 API/database tests, 95 service integration tests
and five real Docker deployment tests. Migration upgrade/downgrade and metadata
agreement are included. Raw verification logs, generated credentials and machine
paths remain in ignored local artifacts.

A separate Compose project also passed startup from empty volumes, migration to
`0004`, automatic upload-expiry configuration, API readiness and clean shutdown.
Its temporary containers and volumes were removed without touching the main stack.
