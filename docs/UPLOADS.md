# Image uploads and thumbnails

An authenticated client now uploads directly to MinIO, completes registration
through FastAPI, and reads the resulting image through private, temporary URLs.
PostgreSQL stores the image's project, dimensions, SHA-256 and object keys.
A local Celery CPU worker generates its thumbnail through Redis.

## Run locally

Update the dedicated API environment, start the Docker services, and configure
temporary-object expiry:

```bash
make api-env UV="$HOME/.local/share/visionforge/tools/bin/uv"
make up
make migrate
make storage-config
```

Run these in three terminals from the repository root:

```bash
make api
make worker
make beat
```

The worker consumes only the `cpu` queue with two processes. Beat is a separate
scheduler: every 60 seconds it queues a recovery scan for images whose `thumb_key`
is still null. Run one scheduler for this environment. Its local schedule file
is ignored under `artifacts/celery/`. Stop each process with Ctrl+C.

API and worker use the same small Python environment, without PyTorch. Docker
runs PostgreSQL, Redis and MinIO; API/worker container packaging is still pending.
The GPU host is not involved in uploads or thumbnails.

Existing `.env` credentials work. `REDIS_HOST` defaults to loopback.
`S3_ENDPOINT_URL` is the server's storage address; `S3_PUBLIC_ENDPOINT_URL` is
the browser's address. Both default to `http://127.0.0.1:${MINIO_API_PORT}`.
They are separate because a container may need `http://minio:9000` while a browser
needs the published host address. Never rewrite the host of an already signed URL.
`S3_REGION` defaults to `us-east-1`, and `S3_IMAGES_BUCKET` to `vf-images`.
Local development uses the existing MinIO credentials.

The API and MinIO allow browser origins `http://localhost:3000` and
`http://127.0.0.1:3000`. API origins can be overridden with the JSON list
`CORS_ORIGINS`; MinIO's list is configured in Compose. Buckets remain private.

## Upload flow

First create a project and obtain an access token using [API.md](API.md) and
[PROJECTS.md](PROJECTS.md). All API routes below require the bearer access token.

1. Calculate the file's byte length and lowercase hexadecimal SHA-256 locally.
2. `POST /api/v1/projects/{project_id}/uploads/presign` with:

   ```json
   {
     "files": [
       {"content_type": "image/png", "size": 1234, "sha256": "<64 lowercase hex characters>"}
     ]
   }
   ```

   Use the real size and digest. The batch accepts 1–20 files. The response is an
   array in request order; each entry has `upload_url`, `upload_token`, `headers`
   and `expires_in` (900 seconds).
3. PUT the raw file bytes to `upload_url`, sending the returned headers. Do not use
   multipart/form-data or send the API bearer token to MinIO. Send the exact file
   body so its automatically generated Content-Length matches the signed length.
   For a browser, use a `File` or `Blob` as the request body.
4. Complete each file separately with
   `POST /api/v1/projects/{project_id}/uploads/complete`:

   ```json
   {"upload_token": "<the token returned for this file>"}
   ```

   The response is 201 for a new image or 200 for an existing image with the same
   SHA-256 in that project. Completion can be retried within the grant's lifetime.
5. Poll `GET /api/v1/images/{image_id}` until `thumbnail_url` is non-null.
   Original and thumbnail GET URLs expire after 300 seconds; call the API again
   to obtain fresh URLs.

For command-line PUTs, after assigning the returned values to shell variables:

```bash
curl --fail-with-body --request PUT "$UPLOAD_URL" \
  --header "Content-Type: image/png" \
  --header "x-amz-checksum-sha256: $CHECKSUM_HEADER" \
  --data-binary @image.png
```

`CHECKSUM_HEADER` is the returned base64 checksum header, not the hexadecimal
digest used in the presign request. These URLs and grants are temporary bearer
capabilities; the client must keep them private.

## Validation and storage

Accepted files are single-frame JPEG/PNG, RGB or grayscale, at least 10×10 pixels,
at most 20 megapixels and at most 20 MiB. The API decodes the file, verifies its
actual format, length and SHA-256, and rejects truncated files, transparency,
palette images, animation and non-default EXIF orientation. Normalize orientation
before uploading. This matches the current detector preparation rules and avoids
different coordinates between labels, previews and training.

The signed PUT binds content type, length and SHA-256. The separate upload grant
binds the owner, project and server-generated temporary key; clients cannot choose
object keys. Completion checks ownership again and registers verified metadata.
A parent-project row lock and the existing `(project_id, sha256)` uniqueness
constraint handle concurrent completion. Another project can store the same bytes.

Objects use these prefixes inside the private image bucket:

- `uploads/{project_id}/{grant_id}`: temporary uploads.
- `{project_id}/originals/{sha256}.png` or `.jpg`: verified originals.
- `{project_id}/thumbs/{sha256}-v1.jpg`: 320-pixel maximum-edge JPEG previews.

The browser never sends file bytes through FastAPI. **Completion does read the
bounded object from MinIO for verification and writes a verified original.**
This adds backend I/O and request time, but ensures database metadata matches the
actual image and prevents a reusable upload URL from changing a registered original.
Moving verification into an asynchronous ingestion job is a later scaling option.
See [S3 presigned requests](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/generate_presigned_url.html).

Successful completion removes its temporary object. `make storage-config`
preserves unrelated lifecycle rules and makes remaining `uploads/` objects
eligible for expiry after one day; physical cleanup is asynchronous. Registered
objects are outside that prefix. PostgreSQL and S3 have no shared transaction:
a crash after writing an original but before committing its row can leave an
unreferenced original. Retry reuses its deterministic key; broader orphan cleanup
and populated-project deletion remain pending.

## Worker recovery and gallery reads

Completion queues thumbnail work after committing the row. If publishing fails,
registration still succeeds and the thumbnail remains pending. Beat's recovery
scan finds these rows and requeues them. The database is the source of pending
work; no volatile in-process background task is responsible for recovery.

Thumbnail tasks verify the original hash, generate a preview and save `thumb_key`.
Repeated delivery is safe. Storage/database failures get bounded task retries;
the periodic scan also revisits unfinished rows. Permanently missing or corrupted
originals remain pending and require repair. There is no separate thumbnail-error
status yet. Celery's [periodic-task guide](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html)
explains the worker/scheduler split.

`GET /api/v1/projects/{project_id}/images` returns `items`, `page`, `page_size` and
`total`, newest first with UUID as a tie-breaker. Optional filters are
`status=unlabeled|labeled|in_review` and `class=<class UUID>`. Page size is 1–100.
Every item includes private original/thumbnail URLs; another owner's image or
project returns 404. Class filtering uses the current annotation records;
[annotation reads and saves](ANNOTATIONS.md) now update those records through HTTP.

## Tests

`make test-db` still needs only PostgreSQL. `make test-uploads` additionally uses
real MinIO and Redis, a disposable database and bucket, an isolated Redis key
prefix, and a real Celery consumer. It cleans up only its own test resources.
Tests cover signed PUTs, byte tampering, formats and limits, expired/scoped grants,
deduplication and races, private reads, gallery filters, CORS, lifecycle rules,
storage failures, worker delivery and recovery after a simulated publish failure.

Verified on 2026-10-20: 10 storage/queue integration tests, 31 API/database tests
and 110 ML/setup tests pass, along with lint. The storage/queue tests use local
Docker PostgreSQL, MinIO and Redis; thumbnail work runs in a real Celery consumer.
