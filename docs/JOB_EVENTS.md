# Live training events

`GET /api/v1/jobs/{job_id}/events` streams progress for the authenticated project
owner as `text/event-stream`. It connects the existing worker and job records to
the future training UI; charts and the web client are still separate work.

The path is: GPU supervisor → PostgreSQL commit → Redis notification → local API
→ browser SSE. The worker publishes to `job:{job_id}:events`, as specified in the
build plan. Redis wakes the API promptly. The API reads committed status and epoch
metrics from PostgreSQL before sending them, so delayed or duplicate notifications
cannot overwrite newer state or invent metrics. If Redis is unavailable or a
notification is lost, the API checks PostgreSQL every two seconds.

This uses the existing pinned dependencies, with no schema migration. Restart the
API and update the GPU supervisor source to enable notifications. Existing workers
still provide status and metrics through the database fallback.

## Authentication and connection

Use the same access-token bearer header as the REST endpoints. Missing, invalid,
expired or refresh tokens return 401. Missing and foreign jobs both return 404.
Do not put tokens in URLs. Native browser `EventSource` does not provide a custom
Authorization header; the frontend should use a streaming `fetch` request:

```javascript
const headers = { Authorization: `Bearer ${accessToken}` };
if (lastEpochEventId) headers["Last-Event-ID"] = lastEpochEventId;
const response = await fetch(`/api/v1/jobs/${jobId}/events`, {
  headers,
  signal: abortController.signal,
});
// Check response.ok, then decode response.body and parse complete SSE frames.
// A network chunk may contain half a frame or several frames.
```

CORS allows `Last-Event-ID` on the configured frontend origins. Responses use
`Cache-Control: no-store` and `X-Accel-Buffering: no`; configure any future reverse
proxy to preserve streaming and allow idle periods longer than 15 seconds.

Authentication and project ownership are checked again at each database refresh.
If access changes after headers were sent, the stream emits a safe `error` event
and closes. It cannot change an already-started HTTP response to 401 or 404.
Refresh tokens through the existing authentication endpoint before reconnecting
when `reason` is `authentication_required`.

## Event contract

Every JSON event has `type`, `job_id` and a UTC ISO timestamp `ts`. SSE's `event:`
field matches `type`. The timestamp is the time the event is emitted, including
when old metrics are replayed; it is not the original epoch completion time.

| Event | Payload and client behavior |
| --- | --- |
| `status` | Current `status`, `gpu_id`, cancellation/start/finish timestamps, safe `error` and successful `model_id`. Replace the displayed state. |
| `epoch` | One complete epoch's numeric `metrics`. Upsert chart points by job, epoch and metric key. |
| `log` | A fixed, safe progress `line` for dataset checking, ML execution or artifact upload. Append to the progress view. |
| `done` | Terminal state with the same fields as `status`. Stop reconnecting. All remaining epochs precede this event. |
| `reconnect` | `reason: "stream_window_ended"`. Reopen the stream with the last processed epoch ID. |
| `error` | `reason` is `authentication_required`, `job_not_found`, `database_unavailable` or `stream_unavailable`. Handle the reason and close the reader. |

A metric frame looks like:

```text
id: epoch:<job-uuid>:7
event: epoch
data: {"type":"epoch","job_id":"<job-uuid>","ts":"2026-10-21T12:00:00+00:00","epoch":7,"metrics":{"train_loss":0.21,"val_accuracy":0.94,"lr":0.0003}}

```

Metric keys use `<split>_<name>` from the stored REST metrics; classifier
`train/learning_rate` becomes `lr`. Examples include `train_loss`, `val_accuracy`,
`val_fitness`, `val_mAP50(B)` and `train_lr/pg0`. Detector punctuation remains part
of the JSON key. This names metrics consistently without converting accuracy to
percentages or changing their values.

Progress lines are intentionally curated. Raw subprocess logs can contain file
paths, credentials in exception messages or other private diagnostics, so those
files remain on the worker. Log messages are best-effort Pub/Sub notifications:
they are not persisted, replayed or used as authoritative job state. Detailed
private log viewing and a web log panel remain separate work.

## Reconnect and recovery

A new stream sends the current status, then existing epochs in ascending order,
then live updates. A terminal job sends its remaining metrics followed by `done`
and closes. Status events are sent only when public state changes; the worker's
10-second database heartbeat does not create repeated status events.

Save an epoch frame's `id` **after** applying it to the chart. On reconnect, send
that exact value in `Last-Event-ID`. The API replays epochs strictly after it.
Status, log and terminal events do not advance the metric cursor. A cursor for a
different job, a malformed cursor, or one beyond the recorded history returns 422.
On a fresh page with no cached chart data, omit the cursor to rebuild the history.

The worker commits all metrics for a completed epoch together and appends later
epochs without changing earlier ones. Epoch-based replay relies on that invariant;
manual partial inserts or backfilling already-emitted epochs are unsupported.
The client should upsert chart points because reconnecting without a saved cursor
can legitimately replay data it has already displayed.

Redis Pub/Sub does not retain events. PostgreSQL therefore handles replay and
missed notifications; Redis is only the low-latency path. A worker stops trying to
publish notifications for its current job after its first Redis error, while
training and durable metric writes continue. New jobs and stream connections try
Redis again. This avoids an event delivery failure turning a successful training
run into a failed job.

Cancellation and heartbeat recovery updates also reach connected clients through
the database check. A recovered job reports `failed` with unknown execution state,
not an assertion that its remote process has stopped. See [worker recovery](GPU_WORKER.md#heartbeats-and-recovery).

## Resource limits and tradeoffs

Database reads run off the API event loop and close their sessions before yielding
frames or waiting for Redis. Each read takes at most 20 epochs and 10,000 metric
rows, with a five-second statement timeout. Catch-up drains successive batches
before sending `done`. No database connection is reserved for an idle stream.

Each active stream has its own Redis subscription when available. There are at
most 32 streams per API process; an additional request receives 429 with
`Retry-After: 2`. Streams send keepalive comments about every 15 seconds, advertise
`retry: 2000`, and rotate after roughly five minutes at the next update check.
A disconnected reader releases its subscription and stream slot. The client must
close its reader when navigating away and reconnect with backoff after transport
errors. Slow downstream readers can delay frame delivery and rotation.

The tradeoff versus a durable event table or Redis Streams is simpler execution
and no new event-retention system, at the cost of database polling per connected
client and no historical log replay. Larger deployments will need shared
subscriptions, stronger connection quotas and public-service rate limits.

## Verification

`make test-uploads` runs SSE checks over a real loopback HTTP server with disposable
PostgreSQL jobs and Redis channels. It verifies live frames, cross-batch replay,
terminal model IDs, access control, expiring credentials, CORS, connection limits,
disconnect cleanup, keepalives, Redis fallback and sanitized database errors.
Worker tests verify real Pub/Sub delivery after database commits and successful
model registration when event publication fails. The orchestration tests use a
simulated ML executor; SSE does not change the training or ONNX algorithms.
