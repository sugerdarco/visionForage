# Deleting populated projects

`DELETE /api/v1/projects/{id}` requires the owner's access JWT. It durably requests
permanent deletion and returns **202**, including for an empty project:

```json
{"id": "PROJECT_UUID", "status": "deleting"}
```

Project list/detail responses expose `deleting: true` while cleanup runs. Poll
`GET /api/v1/projects/{id}` until 404; another DELETE during cleanup returns 202.
A foreign or already removed project returns 404. Project-scoped mutations reject
new work with 409 once deletion is requested. The frontend should disable editing
and display the pending state.

The request cancels queued training jobs and requests cancellation for running
jobs. Every ten seconds, Beat schedules the CPU cleanup worker. Cleanup waits for
running jobs to acknowledge cancellation or for existing lease recovery to finish.
It does not kill unrelated processes or immediately declare an active GPU job gone.
Update and restart remote GPU workers with this revision before using deletion;
they now check project state while registering models and publishing artifacts.

After training stops, cleanup obtains deployment operation locks, stops/removes
owned inference containers, removes their local bundles and prediction outboxes,
and deletes project images, thumbnails, frozen versions, artifacts and prediction
objects. It then deletes dependent database records and the project. Other
projects and object prefixes remain intact. Deleting an active deployment also
intentionally discards its pending prediction outbox.

Failures leave the durable request in place for automatic retry. External cleanup
is idempotent; database metadata is removed only after external cleanup succeeds.
A busy deployment operation or prediction collector defers the attempt. One
project's failure does not prevent attempts for other projects in that batch.
This uses the existing PostgreSQL, Beat and CPU worker instead of another queue
or distributed transaction coordinator. The tradeoff is asynchronous deletion:
partial external cleanup may already have happened while a retry is pending.
There is no undo once deletion starts.

Previously issued temporary upload grants can still write staging objects until
their 15-minute expiry. They cannot register images into the deleted project.
The existing one-day `uploads/` bucket lifecycle removes such late temporary
objects; object-store expiration is not immediate. Presigned reads may remain
usable until their expiry or object removal. Local training scratch/cache files
on remote workers follow the existing worker retention policy; project deletion
purges application storage, not arbitrary remote directories.

`make up` runs migration `0004` and starts the required local services. For native
development, run `make migrate`, `make worker` and one `make beat`. Check CPU worker
logs and service health if deletion stays pending. Cleanup returns deleted,
deferred and failed counts without exposing storage credentials or raw diagnostics.

`make test-uploads` verifies populated cleanup, job cancellation, operation-lock
contention, Docker/storage failures, retries and preservation of another project.
