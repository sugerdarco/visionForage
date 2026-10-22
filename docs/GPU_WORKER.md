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
   the snapshot, then starts the ML Python executable. It renews a PostgreSQL
   heartbeat throughout download, training and publication.
4. The subprocess trains, selects its checkpoint using validation data, evaluates
   the test split, and exports ONNX with the existing parity gates.
5. The supervisor saves epoch metrics to PostgreSQL, uploads verified artifacts
   to private MinIO storage, and atomically registers the model and succeeds the job.

`GET /api/v1/jobs/{id}` now includes `model_id` after successful registration.
Listing jobs and cancelling a completed job return the same model reference.
[Live SSE](JOB_EVENTS.md) now reports status, epoch metrics and safe progress
messages. [Registry browsing](MODELS.md) now exposes registered model metadata;
evaluation reads, downloads, promotion and deployment remain separate work.

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
After committing epoch metrics, the worker publishes a best-effort Redis event.
The [SSE endpoint](JOB_EVENTS.md) streams committed metrics and can replay them
after reconnecting; the existing paginated metrics endpoint remains available.
Redis publication failure does not fail training.

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

## Heartbeats and recovery

Each claim creates an internal random ownership token and a database-clock
heartbeat. A separate thread renews that heartbeat every 10 seconds, including
while a storage call or an ML epoch is taking time. Metric writes, heartbeat
renewals and model registration require the same token and an unexpired
180-second lease. An expired worker cannot renew itself, even before recovery runs.
Failure updates also require the token, preventing an old process from overwriting
a different claim. These internal fields are not exposed in job responses.

The local CPU dispatch task checks for expired running jobs every 10 seconds,
processing at most 100 per pass with row locks and `SKIP LOCKED`. It marks them
`failed` with `Worker heartbeat expired; execution state unknown`. This includes
jobs with an unacknowledged cancellation request: losing contact cannot prove that
the remote process stopped, so recovery does not report them as cancelled or claim
that the GPU is free. Scheduler delays, CPU queue backlog and database outages can
delay reconciliation. Keep one Beat scheduler and the CPU worker running.

The supervisor stops ML execution if a control check or heartbeat fails. GPU-worker
PostgreSQL connections use a 15-second statement timeout, 10-second lock timeout,
and TCP keepalive/user-timeout settings to bound broken control connections.
A local watchdog also rejects a heartbeat that has not succeeded for 90 seconds.
During storage I/O, stopping still waits for the current bounded operation to
return. A late upload can leave an orphan object, but it cannot register a model
after recovery or lease expiry.

We use database ownership rather than Celery visibility timeout as the execution
authority: a Redis redelivery does not tell us whether an earlier GPU process is
still running. The three-minute lease allows scheduling jitter and separates
worker liveness from epoch duration. The tradeoff is delayed crash detection;
a transient control-connection failure also stops work conservatively. Failed
jobs are **not automatically retrained**. Inspect the private worker and confirm
that its process stopped before submitting a new job. Resuming checkpoints and
an operator retry endpoint remain future work.

The dispatcher also republishes unclaimed queued jobs after five minutes, with
the same task ID. This repairs confirmed messages lost from Redis. An offline or
busy GPU queue can accumulate duplicates at one per queued job per five minutes;
messages are not deleted or assumed to be unique. The atomic database claim keeps
those deliveries from starting the same job twice. Stop submitting large backlogs
when no GPU workers are available; queue compaction is not implemented.

### Updating existing installations

Stop or drain old GPU supervisors and confirm their ML children have exited before
upgrading: old code does not enforce the new ownership checks. Stop CPU workers
and Beat, pull the same code on both machines, apply `make migrate` locally, then
restart CPU workers, Beat and the new GPU supervisors. Migration `0003` adds the
two nullable ownership columns and updates the dispatch/recovery indexes; existing
records are preserved. It gives existing running rows a fresh three-minute recovery
window. Those rows cannot be resumed by the new worker and become failed after
that window unless already reconciled. Do not reset them blindly to queued.

## Remaining operational work

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
Recovery tests also cover lost confirmed queue messages, heartbeat expiry,
background renewals, changed ownership, concurrent recovery, late uploads and
termination of a real child process when its job is recovered.
These tests do not claim that their synthetic byte artifacts are trained models.
The existing ML suite separately checks real optimization, evaluation and ONNX.

### Real queued verification, 2026-10-22

Both outstanding worker checks passed on source revision
`a872639fba1553391fadee8f5c1aca0f409b1f87`: a newly queued detector registered
under the unchanged export gate, and real ML processes ran concurrently on two
allocated GPUs. The [sanitized result snapshot](results/gpu-worker-verification-20261022.json)
records job outcomes, timing, parity and artifact SHA-256 values.

| Job | GPU | Completed epochs | Claim to registration | Sampled peak process memory |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8s detector | 4 | 97 | 20 min 14 sec | 4,904 MiB |
| ResNet18 classifier 1 | 5 | 5 | 38.04 sec | 1,312 MiB combined peak for GPU 5 jobs |
| ResNet18 classifier 2 | 5 | 5 | 36.72 sec | Same GPU 5 measurement |

The detector requested 100 epochs with patience 20 and selected epoch 77 using
validation fitness; early stopping ended training at epoch 97. Its frozen snapshot
contained 881 training, 189 validation and 189 test images. Each classifier used
224/48/48 images. All snapshots were created through the platform snapshot builder
from original baseline **training pools only**, repartitioned 70/15/15 with seed 42.
These runs qualify the worker pipeline; they do not replace the established
quality baselines in [RESULTS.md](RESULTS.md).

Each job passed parity on 32 held-out test images at batches 1 and 8, with zero
relative tolerance. The detector's maximum box error was **0.00067138671875 pixels**
against the unchanged **0.002** limit; maximum score error was
**0.0000022351741790771484** against **0.001**. Both classifiers' maximum logit
error was **0.000011235475540161133**, below **0.001**. No exporter or gate code
changed for these runs.

The check used real API job submission, PostgreSQL claims, Redis delivery, remote
GPU supervisors, ML subprocesses and private MinIO artifact publication. The
dispatcher task was invoked directly through Celery's eager execution to publish
messages; GPU consumers received those messages from the real broker. Services
used a disposable database, private test buckets and an isolated shared queue.
One extra delivery per job was published while work was live. All three jobs
registered exactly one model, with 1,358 detector metric rows and 25 rows per
classifier. All nine artifacts per job were downloaded and their stored checksums
verified: **27 artifacts total**.

Monitoring identified only the workers' own ML child processes and captured
**14 samples with both physical GPUs active**, at a nominal two-second interval.
The classifiers ran sequentially on GPU 5 while the detector occupied GPU 4.
This verifies actual overlap and reuse of a worker slot, not cluster-scale load
or an exact memory peak. The two workers used the same work directory and a
limit of two CPU threads per ML process; remote Docker and sudo were unnecessary.

An earlier small detector integration correctly refused registration at a box
error of 0.00225830078125 pixels. That failure remains valid; this successful
fuller training run does not change its result or relax the acceptance criterion.
Before the successful run, one setup attempt failed on an inaccessible cache path
and another was stopped to switch to the newly allocated GPUs. Neither is counted
as a completed training verification.

Temporary workers, their ML children, queue resources, database, buckets and
remote scratch were cleaned up. Model bundles and diagnostics remain private
locally; only the allowlisted result summary is committed. This is an isolated
integration check, not a persistent running deployment.
