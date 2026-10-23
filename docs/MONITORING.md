# Prediction logging and monitoring

New deployments record prediction images in private MinIO storage and results in
PostgreSQL. Owner-scoped APIs expose history and monitoring aggregates. This is
the backend implementation; the browser dashboard and review acceptance workflow
remain frontend/active-learning work.

## Setup

From the repository root, rebuild inference and restart the API, CPU worker and
Beat scheduler with the updated code:

```bash
make inference-image
make gateway
make api
```

Run `make worker` and `make beat` in separate terminals, with one Beat instance.
The existing `make up` bootstrap creates the private `vf-predictions` bucket.
Stop and recreate existing deployments to enable logging; their running containers
retain the old image. Model retraining is unnecessary.

`PREDICTION_LOG_DIR` defaults to `artifacts/predictions`. The API and local CPU
worker must use the same directory and host account. The parent is private
(`0700`). Each deployment gets a writable `/outbox` mount, protected from other
host users by that parent. The model bundle stays read-only.
`S3_PREDICTIONS_BUCKET` defaults to `vf-predictions`. These are operator settings,
not deployment request fields. No new packages or database migration are needed.

## Recording and recovery

The path is inference → local outbox → CPU worker → MinIO + PostgreSQL.

After predicting, inference writes the original JPEG/PNG and JSON event, flushes
them to disk and atomically publishes the event. Its response adds `prediction_id`,
equal to the existing `request_id`. This identity is durable locally before the
response, but may take several seconds to appear in the API. Standalone inference
without logging configuration retains the response without `prediction_id`.

Beat submits collection every five seconds. The CPU worker verifies event/model/
deployment identity, output labels, finite values and image SHA-256, then writes
the image and commits the prediction. It preserves inference time, output,
confidence and latency. Images use `{deployment_id}/{UTC-date}/{prediction_id}.jpg`
or `.png` in the private prediction bucket.

Local files are removed only after the database commit. The event UUID is also
the database primary key; replay after a crash does not insert another prediction.
Object writes are create-only; retries check the existing size and recorded digest.
A local file lock serializes collectors. Storage/database outages leave events
for later scheduled attempts. Stopping a model retains pending logs, which can
still be collected for that stopped deployment.

This keeps database/storage credentials out of inference and avoids a network
callback. The tradeoff is eventual visibility and dependency on the local disk,
CPU worker and scheduler. Back up the outbox with application data if pending
events must survive loss of the host disk.

Each deployment permits at most 1,000 pending image files or 1 GiB of pending
image bytes. A full/unwritable outbox returns `503` with `Retry-After: 5`, rather
than silently returning an unrecorded success. Invalid events remain for local
inspection; the collector reports counts without printing image data or paths.
Incomplete writes from an abruptly killed process are not ingested and can occupy
capacity until inspected and removed while the deployment is stopped. Committed
history has no automatic retention/deletion policy yet.

A response can be lost after recording succeeds. Retrying the HTTP prediction
creates a new prediction; prediction requests themselves are not idempotent.

## Monitoring API

`GET /api/v1/deployments/{id}/stats?window=24h` requires the project owner's access
JWT. Windows are `1h`, `24h` (default), `7d` and `30d`. Foreign/missing deployments
return `404`; unsupported windows return `422`. It needs no Docker connection and
works for stopped deployments.

| Field | Meaning |
| --- | --- |
| `deployment_id`, `window`, `start`, `end` | Identity and UTC range; start inclusive, end exclusive |
| `scope` | `recorded_predictions` |
| `requests` | Recorded predictions within the window |
| `low_confidence_count` | Predictions flagged by the confidence threshold |
| `latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms` | Mean and continuous/interpolated percentiles |
| `class_distribution` | Counts for every class in the model's frozen map |
| `class_count_unit` | `predictions` for classification; `boxes` for detection |
| `confidence_histogram` | Ten bins with `lower`, `upper`, `upper_inclusive`, `count` |

Empty windows return zero counts and `null` latencies. Histogram intervals are
`[0.0,0.1)` through `[0.9,1.0]`, including confidence exactly 1 in the last bin.
Classification counts the winning class once. Detection counts every returned box,
so class totals can exceed requests. No-box predictions still count as requests
and have maximum confidence zero.

Latency includes image decoding, preprocessing, model execution and output
processing. It excludes upload, outbox writes, collection delay, database/storage
work and network transit. It is neither full HTTP latency nor the narrower
model-only ONNX benchmark in the registry.

Aggregates use inference timestamps, not collection time. They exclude rejected
authentication, invalid uploads, busy responses and logging failures. An outage
can temporarily undercount recent predictions until collection catches up. All
aggregates use one PostgreSQL statement/snapshot to keep their totals consistent.

## History and private previews

`GET /api/v1/deployments/{id}/predictions?page=1&page_size=20` returns `items`,
`total`, `page`, `page_size`; maximum page size is 100. Results are newest first,
with UUID as the timestamp tie-breaker. Offset pages can shift as records arrive.

Each item includes `id`, `deployment_id`, `model_id`, `model_version`, `output`,
`max_confidence`, `latency_ms`, `sent_to_review`, `created_at`, `image_url`, and
`image_url_expires_in: 300`. Preview URLs grant private read access for five
minutes; keep them private and request another history page after expiry. Both
routes enforce ownership and return `Cache-Control: no-store`.

Classification maximum confidence is the winning probability. Detection uses the
highest returned box confidence, or zero for no boxes. `sent_to_review` is true
only when this value is strictly below the saved deployment threshold (default
0.6); equality is not flagged. This stores eligibility only; review queue and
acceptance APIs are separate work. The plan's maximum-confidence rule can miss
a low-confidence detection when another box in the same image is highly confident.

## Verification

```bash
make test-inference
make test-db
make test-uploads
make test-deployments
```

Tests cover durable response identity, outbox failure/capacity, private previews,
storage/database recovery, replay after commit, locking, incomplete/invalid events,
real Redis/Celery delivery, stop-before-collection, ownership, pagination, empty
windows, detection counts and exact percentile/histogram boundaries.

The Docker test sends an image through Traefik, stops the model, collects its log
and verifies history, image bytes and statistics. On 2026-10-23, all 32 retained
classifier and 32 detector verification images also matched their reference
predictions, persisted successfully and appeared in monitoring. Their recorded
latencies reproduced the API's p50/p95 values. These are compatibility checks,
not new model-quality or latency benchmarks. Private logs and images stay out of Git.
