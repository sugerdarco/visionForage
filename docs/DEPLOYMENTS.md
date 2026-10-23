# Local model deployments

The control API can deploy a registered model into a local CPU inference container,
report its state and stop it. Promotion and deployment remain separate: promotion
selects the preferred project model; deployment starts a specific model version.
Any owned model from a successful training job can be deployed.

This increment uses dynamically assigned **loopback-only** ports. Traefik routing,
public ingress, browser prediction access, prediction persistence and monitoring
are still pending. The endpoint URL refers to the API machine, so clients must
run there for now.

## Setup

Run the API natively from the repository root, with the existing PostgreSQL and
MinIO services available. The same local account needs Docker CLI/daemon access:

```bash
make inference-image
make api
```

The build pins the Python base image by digest and installs the existing pinned
CPU inference dependencies. Its build context allowlist excludes credentials,
datasets, model binaries and training environments. The API resolves the configured
image tag to a local image ID before launch; it does not pull images automatically.
Rebuilding the tag affects new deployments only.

Optional configuration belongs in the private root `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEPLOYMENT_IMAGE` | `visionforge-inference:0.1.0` | Locally built serving image |
| `DEPLOYMENT_WORK_DIR` | `artifacts/deployments` | Private verified bundle cache |
| `DEPLOYMENT_NAMESPACE` | `visionforge` | Prefix and ownership label for containers |
| `DEPLOYMENT_MAX_ACTIVE` | `2` | Global limit for starting/running deployments |
| `DEPLOYMENT_CPU_THREADS` | `2` | CPU quota and inference thread count per container |
| `DEPLOYMENT_MEMORY_MB` | `1536` | Per-container memory limit, with no additional swap |
| `DEPLOYMENT_READY_SECONDS` | `90` | Readiness polling window after container start |

Use a local Docker context (including local Docker Desktop). Remote Docker contexts
are unsupported because bind mount paths and endpoint loopback addresses must
refer to this machine. Keep the namespace and work directory stable while any
deployment exists. One control installation owns that namespace; do not share it
between independent databases. The API must have write access to the work directory,
which is created with mode `0700`; an existing directory must also be private.

## HTTP contract

All control routes require the owner's access JWT and return `Cache-Control: no-store`.
Unknown or foreign resources return `404`. Container IDs, filesystem paths and
raw Docker/storage errors are not included in responses.

| Method and path | Result |
| --- | --- |
| `POST /api/v1/models/{model_id}/deployments` | Create, prepare and start a deployment; `201` |
| `GET /api/v1/projects/{project_id}/deployments` | Stored deployment history, paginated |
| `GET /api/v1/deployments/{deployment_id}` | Inspect Docker and reconcile this deployment |
| `DELETE /api/v1/deployments/{deployment_id}` | Stop/remove the container and cached bundle; retain history |

Create accepts an empty JSON object or:

```json
{"slug": "casting-line-a", "confidence_threshold": 0.6}
```

Slugs start with a lowercase letter, contain lowercase letters/digits/hyphens and
have at most 63 characters. Omission generates a unique slug. Slugs stay reserved
in history even after stopping; use a new slug for a replacement deployment.
`confidence_threshold` must be a number from 0 to 1. It is stored for future
low-confidence review; it does not currently filter prediction responses.

The response includes `id`, `model_id`, `project_id`, `model_version`, `slug`,
`status`, `confidence_threshold`, `endpoint_url`, `runtime_checked`, `created_at`
and `updated_at`. Only the create response includes `api_key`: a newly generated
secret when running, or `null` if startup failed. Save the key privately; neither
reads nor listings reveal it again. PostgreSQL and the container receive only
its SHA-256 digest. Use the raw key as `X-API-Key` at the returned endpoint with
the [prediction request contract](INFERENCE.md#prediction-contract).

Create is synchronous and can take longer than the readiness window because
artifact downloads and Docker commands happen first. Individual Docker commands
have a 30-second timeout. Do not automatically retry a timed-out POST: it may have
started a container. List deployments and inspect the new record first. If its
one-time key was lost with the response, stop it and create a replacement. There
is no key recovery, rotation endpoint or create-idempotency token yet.

Listings accept `page` (default 1) and `page_size` (default 20, maximum 100) and
return `items`, `total`, `page`, `page_size`. They read PostgreSQL only, so each item
has `runtime_checked: false` and `endpoint_url: null`. Read an individual deployment
for its current endpoint. A read concurrent with an ongoing operation also returns
the last stored state with `runtime_checked: false`.

## How the components connect

1. The API checks project ownership, successful training and the registered artifact
   namespace, sizes and checksums. A PostgreSQL advisory lock serializes capacity
   reservations across API processes; the `starting` record commits before launch.
2. The API downloads only `model.onnx` and `metadata.json` from private MinIO into
   a temporary directory, checks exact byte counts and SHA-256, then renames the
   completed bundle into place. Graphs are limited to 1 GiB and metadata to 1 MiB.
3. Docker mounts that bundle read-only. The container receives model identity,
   hashes, slug, key digest and CPU thread count. It receives no storage, queue or
   database credentials and independently verifies the bundle before becoming ready.
4. The API waits for Docker's `/readyz` health check, stores the container ID and
   marks the deployment `running`. Predictions go directly to the inference port.
5. Stop checks all ownership labels, removes the inspected container by ID,
   confirms its absence and deletes the local bundle before marking it `stopped`.
   Repeating stop is safe; immutable artifacts remain in MinIO.

Per-deployment PostgreSQL locks serialize launch, reconciliation and stop. A stop
while another operation is active returns `409`; retry after it finishes. Capacity
exhaustion, a duplicate slug or a concurrent capacity reservation also returns `409`.

If startup fails and cleanup succeeds, create returns `201` with a durable `failed`
record and no key. If Docker is unavailable or removal cannot be confirmed, the
API returns `503` and retains the previous state/capacity reservation. It does not
claim a container stopped when its state is unknown.

There is no background deployment reconciler yet. An individual GET recovers a
healthy container left in `starting` after an API interruption, or cleans up a
missing/exited/unhealthy deployment and marks it `failed`. A fresh starting record
gets a grace period of `DEPLOYMENT_READY_SECONDS + 30` seconds before failed-start
cleanup. DELETE can clean up an interrupted start directly. A failed or stopped
deployment is not automatically restarted. During a Docker outage, listings still
work; reads/stops that require Docker return `503`.

## Choices and resource limits

The Docker CLI uses the existing local installation and avoids another SDK
dependency. Commands use argument arrays, fixed options and timeouts. Each
container has deterministic naming plus scope/deployment/model/project labels;
the API verifies all four before starting or removing it.

Containers run as UID/GID `10001`, with a read-only root filesystem, all Linux
capabilities dropped, no-new-privileges, 128 PIDs and a 64 MiB temporary filesystem.
Docker logs rotate at 10 MiB with two files. Temporary logs disappear when the
container is removed. The default two-deployment limit bounds model memory and
CPU use on a small local Docker VM; other services still need their own headroom.
Container networking is Docker's default bridge with a loopback-only published
port; outbound network isolation is not implemented.

The image retains CPU PyTorch/torchvision for the classifier's exact bicubic
preprocessing. That costs disk and startup time, but preserves training/serving
agreement; detector inference uses ONNX Runtime, OpenCV and NumPy. Neither task
needs a GPU for serving. Synchronous startup keeps this increment small; a durable
background deployment worker would make longer downloads and cancellation easier.

## Verification

```bash
make test-db           # PostgreSQL state, ownership, capacity, race and error checks
make test-deployments  # Opt-in: built image + local Docker, PostgreSQL and MinIO
```

The Docker suite creates disposable databases, buckets, directories and uniquely
labeled containers. It checks a real prediction from a tiny ONNX fixture, runtime
restrictions, normal stop, exited-container reconciliation and invalid-bundle
cleanup. A modified storage object is rejected before container creation. The
suite removes only its own resources.

On 2026-10-23 the retained trained classifier and detector were also uploaded into
disposable private artifact storage and deployed through these APIs, one at a time.
All 32 classifier predictions matched the saved reference probabilities within
absolute `1e-7`; all 32 detector predictions matched box counts/classes and normalized
boxes/scores within `1e-6`. Both deployments stopped and cleaned up afterward. This
checks serving compatibility, not a new quality or latency benchmark. Raw test
logs, endpoint keys, artifact paths and binary models remain outside Git.
