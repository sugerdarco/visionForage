# Training worker

The worker connects [training jobs](TRAINING_JOBS.md) to the existing ResNet18 and
YOLOv8s training, held-out evaluation and ONNX export functions. Local Docker
still runs PostgreSQL, Redis and MinIO. A remote Linux Python process consumes
the `gpu` queue and launches one isolated ML subprocess on its configured GPU.
The remote account needs neither sudo nor Docker access.

The connection is:

1. The API commits the job and its frozen dataset version ID to PostgreSQL.
2. The local CPU dispatcher publishes the job ID to Redis.
3. The remote supervisor claims the matching queued job, downloads and verifies
   the snapshot, then starts the ML Python executable.
4. The subprocess trains, selects its checkpoint using validation data, evaluates
   the test split, and exports ONNX with the existing parity gates.
5. The supervisor saves epoch metrics to PostgreSQL, uploads verified artifacts
   to private MinIO storage, and atomically registers the model and succeeds the job.

`GET /api/v1/jobs/{id}` now includes `model_id` after successful registration.
Listing jobs and cancelling a completed job return the same model reference.
Registry browsing, promotion, live SSE and deployment remain separate work.

## Two Python environments

Use the existing CUDA ML environment unchanged. Create a separate service
environment using the already pinned API requirements; this provides Celery,
PostgreSQL, S3, Pillow and validation without installing another copy of PyTorch.
Reusing that lock keeps service versions identical across the API and workers;
the tradeoff is a few API-only packages in the supervisor environment.

From a remote checkout, using your own absolute paths:

```bash
export RUNTIME_DIR=/var/tmp/visionforge-user
"$RUNTIME_DIR/tools/bin/uv" venv \
  --python "$RUNTIME_DIR/venv-cu126/bin/python" "$RUNTIME_DIR/venv-worker"
"$RUNTIME_DIR/tools/bin/uv" pip sync \
  --python "$RUNTIME_DIR/venv-worker/bin/python" requirements/api.txt
```

Use a fresh environment path on first setup. On later updates run only `pip sync`
for that service environment. Do not sync API requirements into the ML environment.
The supervisor preserves the ML executable's virtualenv symlink: resolving it to
the base interpreter would lose the virtualenv's installed packages.

The implementation uses Linux process groups, `flock` and `prctl`. Python 3.11,
the pinned CUDA stack, GPU access and ordinary user-owned directories are sufficient;
no system service installation or container runtime is required remotely.

## Private service connection

Keep local service ports on loopback. For a workstation/VM setup, run SSH reverse
forwards from the workstation in a dedicated terminal:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:15432:127.0.0.1:5432 \
  -R 127.0.0.1:16379:127.0.0.1:6379 \
  -R 127.0.0.1:19000:127.0.0.1:9000 \
  user@gpu-host
```

These are placeholder connection details. Replace the account/host and ports with
your private configuration. SSH forwarding must be permitted by the host. A remote
process then reaches PostgreSQL at its own loopback port 15432, Redis at 16379 and
MinIO at 19000. This avoids opening the workstation services to the network; the
tradeoff is that the workstation, Docker and SSH connection must remain available.

Create a private `.env` in the remote checkout, with permissions `0600`. Copy only
the required service credentials from the local private configuration, using a
private transfer method. The supervisor does not need `JWT_SECRET`. Never commit
this file, connection transcripts, or worker logs.

```dotenv
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=15432
POSTGRES_DB=visionforge
POSTGRES_USER=visionforge
POSTGRES_PASSWORD=<private database password>
REDIS_HOST=127.0.0.1
REDIS_PORT=16379
REDIS_PASSWORD=<private Redis password>
MINIO_ROOT_USER=visionforge
MINIO_ROOT_PASSWORD=<private MinIO password>
S3_ENDPOINT_URL=http://127.0.0.1:19000
S3_PUBLIC_ENDPOINT_URL=http://127.0.0.1:19000
S3_IMAGES_BUCKET=vf-images
S3_DATASETS_BUCKET=vf-datasets
S3_ARTIFACTS_BUCKET=vf-artifacts
WORKER_ML_PYTHON=/var/tmp/visionforge-user/venv-cu126/bin/python
WORKER_WORK_DIR=/home/user/visionforge/worker-jobs
WORKER_CACHE_DIR=/home/user/visionforge/models
WORKER_GPU_ID=<allocated physical GPU index>
WORKER_DEVICE=cuda
WORKER_CPU_THREADS=2
WORKER_TIMEOUT_SECONDS=86400
```

Replace every placeholder before starting. Keep `WORKER_WORK_DIR` private and
outside tracked source, or under the ignored `artifacts/` directory locally.
Classifier weights use the configured cache root; pinned detector weights use its
`yolo/` subdirectory. A missing weight may require an upstream download. For a
local CPU check, set `WORKER_DEVICE=cpu` and point `WORKER_ML_PYTHON` at the CPU
environment; all dataset and export gates still apply.

Start the supervisor from the checkout:

```bash
make gpu-worker API_PYTHON="$RUNTIME_DIR/venv-worker/bin/python"
```

The launcher uses Celery's `solo` pool, concurrency one and only the `gpu` queue.
CUDA runs in a fresh subprocess, avoiding CUDA initialization in a forked Celery
process and isolating Ultralytics settings between jobs. The subprocess has
`CUDA_VISIBLE_DEVICES` set to the configured physical index and uses zero data
loader workers. Password/secret/token environment variables are removed before
launch; the ML code does not need service credentials.

For additional allocated GPUs, start one supervisor per physical index by
overriding `WORKER_GPU_ID` in its environment. Use the **same work directory** for
supervisors on the same machine: its per-GPU file lock prevents overlapping local
jobs on a configured GPU. This lock does not reserve hardware against other users
or programs. Allocate GPUs before starting workers; current free memory alone is
not an allocation. Multiple workers consume the shared GPU queue as capacity
becomes available, without a client selecting the worker.

## Verification, metrics and publication

The worker reads `DatasetVersion`, never live image or annotation rows. It verifies
the manifest and audit SHA-256 values, project namespaces, task, class map and
split counts. Every original must match its byte checksum, decoded dimensions and
RGB pixel checksum. Identical pixels cannot cross splits. Existing snapshot limits
remain: 10,000 images, 1,000 classes, 1 GiB of originals, one billion decoded pixels,
and 32 MiB per JSON document. Classification needs at least two classes.

Training uses the stored hyperparameters and pretrained baseline weights. The
subprocess writes atomic epoch history files; the supervisor polls them and saves
numeric metrics under the job. Classification records train/validation loss and
accuracy plus learning rate. Detection records loss components, validation metrics,
fitness and learning rates. Duplicate metric writes preserve the original values.
The existing metrics endpoint can be polled until SSE is added.

Successful training alone is insufficient for registration. Test evaluation and
ONNX export must complete, with opset 17, dynamic batches 1 and 8, and 32 distinct
test image keys. Classification uses absolute tolerance 0.001. Detection keeps the
approved 0.002-pixel box tolerance and 0.001 score tolerance; both use zero relative
tolerance. The supervisor rechecks provenance, hashes, class maps and gate reports.
There is no tolerance relaxation or fallback model when a gate fails.

Each successful model has nine private artifacts: the manifest, audit, weights,
training report, epoch history, test evaluation, ONNX model, export metadata and
verification report. Each object uses a unique job/attempt namespace and a
create-only write, with a checksum and size recorded in `Model.metrics.artifacts`.
Individual files are capped at 1 GiB. Model metrics also include `test`, `parity`
and `latency`. Class maps retain the frozen name-to-index mapping. Latency is
measured on the worker host's CPU and must not be presented as workstation latency.
Export stays on the worker host to keep checkpoints and verification together.
The tradeoff is that its GPU slot remains occupied while CPU export finishes;
splitting export onto another worker is a later throughput optimization.

After all uploads, the worker locks the project and job, checks cancellation again,
allocates the next project model version, inserts the model, and marks the job
`succeeded` in one PostgreSQL transaction. Concurrent jobs cannot share a model
version. The new model is not automatically promoted or deployed.

## Cancellation and failure behavior

The supervisor claims a job only while it is `queued`, uncancelled, and its stored
Celery task ID matches the message. Duplicate, stale and already-cancelled messages
do not start work. PostgreSQL performs the claim; a Celery task ID alone is not a
deduplication guarantee.

During ML execution the supervisor checks PostgreSQL about once a second. A cancel
request, control-check error, subprocess failure, timeout, or oversized log stops
the process group. It sends SIGTERM, waits up to five seconds, then SIGKILL. The
direct ML subprocess also installs a Linux parent-death SIGKILL, covering abrupt
supervisor death. While downloading or uploading, cancellation is checked between
storage operations and can wait for the current operation to return. The configured
timeout applies to the ML subprocess, and its log limit is 16 MiB.

The job becomes `cancelled` only after execution stops and the worker acknowledges
the request. Success registration and cancellation serialize on the job row. Failed
training, invalid data, failed parity or incomplete uploads produce no registered
model. API error text is a safe summary; detailed diagnostics stay in private
`WORKER_WORK_DIR/<job-id>.log` files. Temporary data, checkpoints and exports are
removed when a normal task finishes; logs remain for operator inspection.

## Remaining operational work

This is not automatic recovery from every infrastructure failure. Abrupt worker
death or a database outage that prevents the final state update can leave a job
`running`. Duplicate delivery will not restart it, avoiding overlapping training,
but leases, heartbeats and stalled-job reconciliation remain to be implemented.
Confirm that its subprocess is stopped before manually reconciling such a job.
Redis message loss after confirmed dispatch has the previously documented recovery
gap. Do not delete queues or blindly reset running jobs to queued.

MinIO and PostgreSQL do not share a transaction. Partial uploads or cancellation
after upload can leave unreferenced artifact objects; automatic orphan cleanup,
log retention and scratch cleanup after a machine crash are still pending. The
current private deployment also reuses existing service credentials; narrower
worker database/storage credentials are future deployment hardening.

## Checks

`make test-uploads` exercises real PostgreSQL and MinIO with an explicitly simulated
ML executor for failure and concurrency cases. It covers single registration,
duplicate delivery, cancellation, immutable artifact writes, corrupted data,
failed export, partial upload, metric replay and actual subprocess termination.
These tests do not claim that their synthetic byte artifacts are trained models.
The existing ML suite separately checks real optimization, evaluation and ONNX.

A real GPU classification integration check completed training, 32-image ONNX
verification, private artifact upload and model registration. Detector integration
completed training and test evaluation on a small NEU-DET subset, then correctly
refused registration because its maximum box difference was 0.00225830078125 pixels,
above the unchanged 0.002 limit. A successful newly queued detector registration
under that gate is still unverified. These small integration runs do not replace
the established quality baselines in [RESULTS.md](RESULTS.md).
