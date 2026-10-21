# Training requests and queue handoff

Training requests now connect a frozen dataset version to a durable job record.
The local API validates the request and stores it in PostgreSQL. A local CPU
worker publishes the job ID to Redis's separate `gpu` queue. No model, image,
credential or remote connection string travels in the task payload.

**Current boundary:** submission, listing, metrics reads, cancellation requests
and dispatch are implemented. The GPU consumer, frozen-data download,
train/evaluate/export pipeline, metric writes, model registration and live SSE
remain pending. Jobs stay `queued` until that consumer is implemented and started;
a dispatch timestamp does not mean training has begun. Existing standalone ML
scripts remain available for training.

## Run locally

After the [local services](LOCAL.md) are running, apply the additive migration:

```bash
make migrate
make test-db
make test-uploads
```

Run these in separate terminals:

```bash
make api
make worker
make beat
```

Use one Beat scheduler. It sends `visionforge.training.dispatch` every 10 seconds
to the CPU queue, alongside the existing thumbnail recovery schedule. The CPU
worker explicitly consumes only `cpu`; keep the `--queues=cpu` setting in
`make worker`. Starting Celery without an explicit queue selection can make it
consume other queues known to that app and discard unregistered task types.

Migration `0002` adds nullable `dispatched_at` and `cancel_requested_at` columns
and an index for pending dispatch. Existing records remain intact. Legacy rows
without a Celery task ID are not dispatched automatically.

## HTTP contract

All paths below start with `/api/v1` and require the owner's bearer access token.
Foreign and missing records both return 404. Responses use `Cache-Control: no-store`.

| Method and path | Behavior |
| --- | --- |
| `POST /projects/{id}/jobs` | 201: persist a new queued job and its effective hyperparameters |
| `GET /projects/{id}/jobs` | Jobs newest first; `page=1`, `page_size=20`, maximum 100 |
| `GET /jobs/{id}` | Status, parameters, dispatch/start/finish/cancellation timestamps and error |
| `GET /jobs/{id}/metrics` | Stored metrics in epoch/split/name order; `page_size=100`, maximum 100 |
| `POST /jobs/{id}/cancel` | Cancel queued work, or request cooperative cancellation of running work |

Classification example:

```json
{
  "dataset_version_id": "<frozen-version-uuid>",
  "model_arch": "resnet18",
  "hyperparams": {"epochs": 20, "batch_size": 64}
}
```

Use `yolov8s` for a detection project. Only these two implemented baseline
architectures are accepted; alternative backbones will need matching training
and export support first. Omitting `hyperparams`, or individual fields, fills
and stores the defaults rather than leaving them implicit for a later worker.

| Parameter | ResNet18 default | YOLOv8s default | Accepted range |
| --- | --- | --- | --- |
| `epochs` | 20 | 100 | integer 1–1000 |
| `batch_size` | 64 | 16 | integer 1–256 |
| `patience` | 5 | 20 | integer 1–1000 |
| `seed` | 42 | 42 | integer 0 through 2³²−1 |
| `horizontal_flip`, `vertical_flip` | false | false | booleans |
| `learning_rate` | 0.0003 | unavailable | finite number 1e-8 through 1 |
| `weight_decay` | 0.01 | unavailable | finite number 0 through 1 |
| `label_smoothing` | 0.1 | unavailable | finite number from 0 up to, excluding, 1 |
| `image_size` | fixed by classifier | 640 | integer 64–1280, multiple of 32 |

Unknown fields, numeric strings, booleans in numeric fields and nonfinite values
return 422. Device selection, GPU allocation, worker count, filesystem paths,
shell commands and pretrained-weight policy belong to worker configuration.
Batch-size limits bound requests but cannot guarantee that a given GPU has enough
free memory. Flips stay disabled unless appropriate for the defect task.

The version must belong to the same project and include manifest/audit checksums,
nonempty train/validation splits, and at least **32 test images**. This enforces
the existing ONNX parity sample requirement before expensive training starts.
The API checks stored metadata; the worker must later verify the actual frozen
artifacts and original bytes. It must not substitute current labels or class maps.

Each successful POST creates a separate experiment. There is no submission
idempotency key yet; repeating a POST creates another job. If a response is lost,
inspect the project job list before submitting again.

## State and cancellation

`queued` means accepted but not claimed, including time waiting in Redis.
`dispatched_at` records a successful publish followed by a database commit.
`running` will mean the GPU worker has claimed the job; its `gpu_id` and
`started_at` will identify the allocation and start time.

Cancelling a queued job sets `cancelled`, `cancel_requested_at` and `finished_at`
in one transaction, even if a message is already in Redis. Cancelling a running
job sets only `cancel_requested_at`; it stays `running` until its worker stops
and acknowledges the cancellation. This avoids reporting a GPU as idle while
computation continues. Repeated cancellation preserves timestamps. Cancelling a
terminal `succeeded`, `failed` or `cancelled` job returns it unchanged.

The cancellation endpoint does not kill remote processes or rely on Celery's
in-memory revoke list. The worker must check the durable request between stages
and batches, stop its own work, and prevent a cancelled run from registering a
model. Cancellation and successful model registration must serialize on the job
row so a late cancel cannot overwrite a completed result.

## Why PostgreSQL plus Redis

PostgreSQL is already responsible for jobs, ownership and version references.
Its committed job row also acts as an outbox: the API does not depend on a second
system being available to accept a request. Redis/Celery supplies worker delivery
and routing without putting GPU libraries into the API process.

Each dispatch pass considers at most 100 oldest pending jobs. It locks one row
with `FOR UPDATE SKIP LOCKED`, checks that it is still queued, publishes only its
ID with a stable Celery task ID, then records dispatch. Other dispatchers skip
locked jobs; cancellation uses the same row lock. A broker exception rolls back
that job's dispatch timestamp, leaving it available on the next scheduled pass.
The tradeoff is a short row lock during broker I/O and up to a scheduler interval
of added latency, plus any CPU queue backlog.

PostgreSQL and Redis do not share an atomic commit. If Redis accepts a message
and the database commit fails, the dispatcher sends the same task ID again.
**Celery task IDs do not deduplicate execution.** The GPU consumer must atomically
claim a still-queued, uncancelled row before allocating work and reject duplicate
claims. It must separately handle crashes after claiming; worker leases and
stalled-run recovery are not implemented here.

Once a dispatch timestamp is committed, this dispatcher does not republish it.
Redis data loss or manual queue deletion can therefore leave a job queued with
no message. The current Redis persistence policy is described in [LOCAL.md](LOCAL.md).
Automatic recovery for that case remains part of GPU-worker integration; this
increment only recovers unconfirmed publication. Public job errors should contain
safe summaries, never raw exception strings with credentials or remote paths.

## Validation

The PostgreSQL tests exercise ownership, typed parameters, dataset eligibility,
metrics pagination, concurrent cancellation and terminal-state behavior. The
Redis integration tests use private disposable queues to check CPU-to-GPU routing,
broker failure recovery, bounded batches, concurrent publishers, cancellation
races and duplicate publication after a failed database update. They do not run
a GPU training job or claim that model registration is complete.
