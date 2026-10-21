# Local application services

PostgreSQL, Redis and MinIO run in local Docker Compose. The remote GPU machine
runs training in its existing user-owned Python environment; it needs neither
sudo nor Docker. This follows our available machines rather than the original
plan's assumption that the whole stack runs on the GPU server.

| Component | Stores / does | Why this choice |
| --- | --- | --- |
| PostgreSQL 16 | Users, projects, labels, job status and metrics | Transactions and relationships fit this data; a shared server supports API and worker processes better than a local SQLite file. |
| MinIO | Images, frozen manifests, weights and ONNX files | The S3 interface supports direct uploads and future S3 storage. Large binaries stay outside database rows. |
| Redis 7 | Future Celery queue and live job events | Fits the planned Celery worker and event stream. PostgreSQL remains the durable source of job status. |

The storage and queue services have health checks and private buckets. The first
database migration is available in [DATABASE.md](DATABASE.md). The local FastAPI
account service is described in [API.md](API.md). [UPLOADS.md](UPLOADS.md) adds
the local CPU thumbnail worker; GPU training worker integration is still pending.

## One-time setup

Use Linux x86_64 and Docker Compose with `up --wait` support. Docker Desktop works;
the system Docker Engine also works if your account already has access. Commands
use your current Docker context and never change it. Check `docker info` first.

MinIO's community repository is [no longer maintained and distributes source
only](https://github.com/minio/minio). We retain MinIO for the local development
plan, pinning the [2025-10-15 security release](https://github.com/minio/minio/releases/tag/RELEASE.2025-10-15T17-29-55Z).
This does not provide ongoing maintenance: reassess object storage before a
public deployment. The client is pinned to its 2025-08-13 release. Both build
from checksum-verified source archives with Go 1.26.8; no Go installation is
needed on your system. The resulting binaries are Linux amd64 only.

Replace `user@gpu-host` and the example paths with your private SSH configuration.
From the repository root, build on the remote VM (CPU only, four build threads):

```bash
ssh user@gpu-host \
  'bash -s -- /var/tmp/visionforge-user/storage-build' \
  < scripts/build_storage_tools.sh
mkdir -p artifacts/minio
scp 'user@gpu-host:/var/tmp/visionforge-user/storage-build/bin/*' artifacts/minio/
(cd artifacts/minio && sha256sum --check SHA256SUMS)
make env
make up
make check-infra
```

Alternatively, run the same build script locally with an absolute directory and
copy its `bin/` contents into `artifacts/minio/`. The first build downloads a Go
toolchain and dependencies; later runs reuse downloads and the compiler cache.
The local container build only packages the binaries and their licenses on a
pinned Alpine base, verifies their checksums and runs as UID 1000. All binaries
and build caches stay outside Git. PostgreSQL and Redis use official images,
pinned by version and image digest in `infra/compose.yml`.

`make env` creates random independent passwords and an API signing key in `.env`, mode 0600, and never
overwrites an existing file. `.env.example` contains names and nonsecret defaults.
Keep `.env` private; `docker compose config` without `--quiet` expands its secrets.
The credentials are for this local development stack. PostgreSQL's initial user
is an administrator and MinIO uses its root account; application-specific roles
and storage policies belong with the API/worker integration.

`make up` waits for healthy services, then creates four buckets idempotently:
`vf-images`, `vf-datasets`, `vf-artifacts`, and `vf-predictions`. New buckets are
private by default. Object keys will follow the build plan's project/job layout.
Rerunning bootstrap keeps existing data and bucket policies.

## Connections

| Service | From the local host | From future Compose containers |
| --- | --- | --- |
| PostgreSQL | `127.0.0.1:5432` | `postgres:5432` |
| Redis | `127.0.0.1:6379` | `redis:6379` |
| MinIO S3 API | `http://127.0.0.1:9000` | `http://minio:9000` |
| MinIO console | `http://127.0.0.1:9001` | `http://minio:9001` |

Log into the console with `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` from `.env`.
Change the host port variables in `.env` if another local service uses a port.
Container ports remain fixed. Every published port binds **127.0.0.1**, not the
LAN; services communicate through Compose's project network.

The current flow is: browser → API → PostgreSQL for records; browser → MinIO via
presigned URLs for file bytes; API → Redis → local CPU worker for thumbnails.
The future remote training worker will download the frozen manifest and images
from MinIO, call `vf_ml`, upload artifacts, write metrics to PostgreSQL and publish
Redis events.
The API does not import PyTorch. Browser-facing presigned URLs must use the host
endpoint, because browsers cannot resolve the Compose name `minio`.

Remote integration will use reverse SSH forwarding, initiated **from the laptop**:

```bash
ssh -NT -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:15432:127.0.0.1:5432 \
  -R 127.0.0.1:16379:127.0.0.1:6379 \
  -R 127.0.0.1:19000:127.0.0.1:9000 user@gpu-host
```

Then remote clients use `127.0.0.1:15432`, `127.0.0.1:16379` and
`http://127.0.0.1:19000`, with credentials supplied separately. Adjust the
right-hand ports if local mappings change. Forwarding requires the SSH server to
allow it and the remote ports to be free; it needs no sudo. This command is a
connection plan, not a running worker or an automated tunnel supervisor. The
laptop, Docker and tunnel must stay available during integrated jobs. Other
accounts on the shared VM can reach its loopback ports, so authentication still
matters. The standalone baseline runs remain independent of these services.

## Checks and persistence

```bash
make status
make check-infra
# Briefly recreates these three containers to test persistent volumes:
bash scripts/check_local_services.sh --recreate
make down
make up
```

The check creates a uniquely named SQL schema, a Redis key with a ten-minute TTL
and a small object in `vf-artifacts`. It verifies authenticated round trips,
all four buckets, rejection of incorrect database credentials, and rejection of
unauthenticated Redis/S3 access. Cleanup removes only its own probes. The
`--recreate` option verifies the same bytes after replacing the containers.

Named volumes `visionforge_pg_data`, `visionforge_redis_data` and
`visionforge_minio_data` survive `make down`. Do not add `-v` to `compose down`
unless you intend to delete the stored data. Volumes belong to the selected
Docker context; switching contexts does not move them. They are not backups.
Keep the original `.env`: changing `POSTGRES_PASSWORD` does not rotate a password
inside an existing PostgreSQL volume; that requires an explicit database change.

Redis uses append-only persistence with `appendfsync everysec`. This can lose
roughly the last second of writes after a crash, and Pub/Sub events are not
replayed. `noeviction` avoids silently discarding queued jobs when a memory limit
is configured; it does not make the queue an exactly-once delivery system. Later
workers will need retry-safe tasks and durable job status in PostgreSQL.
[Redis persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)
and [PostgreSQL image initialization](https://hub.docker.com/_/postgres) describe
these storage behaviors.

Project and class endpoints are available; see [PROJECTS.md](PROJECTS.md).
Image upload and CPU thumbnail processing are available in [UPLOADS.md](UPLOADS.md).
[Annotation reads and saves](ANNOTATIONS.md) use local PostgreSQL without a worker.
[Dataset imports](IMPORTS.md) use the same local API, storage and thumbnail worker.
[Frozen dataset versions](DATASET_VERSIONS.md) use local CPU validation and the
private `vf-datasets` bucket. [Training jobs](TRAINING_JOBS.md) now use PostgreSQL
for durable requests and the local CPU worker to dispatch to a separate GPU
queue. Remote GPU execution and artifact registration remain pending.
There is no `make up-gpu` or GPU Compose overlay: remote GPU workers will use the
tested Python environment because the remote account has no Docker access.

Verified on 2026-10-19 with Docker Desktop Engine 29.3.1 and Compose 5.6.0:
startup with empty volumes, repeated bootstrap, authenticated reads/writes,
access rejection and persistence across container recreation. All 109 Python
tests and lint passed. Remote worker integration remains a later task.
