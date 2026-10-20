# Projects and class maps

The API now connects an authenticated account to its projects and label definitions.
The access token identifies a user; project queries include that user's ID as the
owner. Class routes first verify ownership of their parent project. These records
will connect image uploads and annotations to the frozen manifests used by training.

Use the [API setup instructions](API.md), run `make api`, and log in through
`/docs`. Authorize with the returned access token. This work runs locally against
Docker PostgreSQL; it does not need Redis, MinIO or the GPU machine yet.

## Routes

All paths below start with `/api/v1` and require an access token.

| Method and path | Behavior |
| --- | --- |
| `POST /projects` | Create a classification or detection project, optionally with an initial class list. Returns 201 and project details. |
| `GET /projects?page=1&page_size=20` | List only the current user's projects. Returns `items`, `page`, `page_size`, `total`. |
| `GET /projects/{project_id}` | Read a project and its classes, ordered by model index. |
| `PATCH /projects/{project_id}` | Rename with `{"name": "New name"}`. Returns updated project details. |
| `DELETE /projects/{project_id}` | Delete a project containing only class definitions. Returns an empty 204 response. |
| `GET /projects/{project_id}/classes` | Read the ordered class list. |
| `POST /projects/{project_id}/classes` | Append a batch of classes atomically. Returns 201 and only the newly added classes. |

Create a project:

```json
{
  "name": "Casting inspection",
  "task_type": "classification",
  "classes": [
    {"name": "ok", "color": "#22c55e"},
    {"name": "defective", "color": "#ef4444"}
  ]
}
```

The server chooses the owner from the token and assigns class indices `0` and `1`.
Clients cannot supply `owner_id`, `project_id` or class `index` in these inputs.
Project detail responses include IDs, timestamps, task type and the class map.
The `classes` field may be omitted or empty when creating a project.

To add more classes, send `{"classes": [{"name": "scratch", "color": "#eab308"}]}`
to the class POST route. That route requires 1–100 classes per request; project
creation accepts at most 100 initial classes. Names are trimmed and must contain
1–200 characters. Class names are case-sensitive and unique within their project.
Colors require six hex digits after `#` and are returned in lowercase.

Only project names can be patched. Task type, owner and class indices are
immutable through these endpoints. Anomaly project creation is deferred with
the plan's anomaly feature, although the database reserves that task value.
Class renaming, deletion and reordering are not exposed.

## Ownership and consistency

Another user's project ID produces the same 404 response as a nonexistent ID,
including for rename, delete and class operations. Missing or invalid access
tokens return 401. Project list counts are owner-scoped too. Authenticated API
responses carry `Cache-Control: no-store`.

Class batches either commit completely or leave the project unchanged. Existing
names return 409; repeated names inside one request return 422. The API holds a
lock on the parent project while finding the next index and inserting the batch.
Two requests for the same project serialize; different projects can proceed
independently. PostgreSQL also enforces unique names and indices. This uses the
existing transaction boundary instead of introducing a Redis lock. See
[PostgreSQL row locking](https://www.postgresql.org/docs/16/explicit-locking.html#LOCKING-ROWS)
and [SQLAlchemy's locking API](https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.with_for_update).

Indices are append-only from the start. Adding a class after a training run keeps
existing indices intact and does not change the trained model's stored class map.
That model still recognizes only the classes it was trained on; using the new
class will require a new dataset version and training run. Future import and
annotation endpoints must preserve this mapping as well.

Project listing uses offset pagination: pages start at 1, page size is 1–100,
and results sort by creation time descending with UUID as a tie-breaker. This
fits a small project dashboard. It is not a frozen snapshot: concurrent creations
or deletions may shift later pages.

Deletion uses one transaction for the project and its class definitions. If
images, annotations, dataset versions or training history still reference them,
foreign keys reject deletion with 409 and class removal rolls back too.
**Populated-project deletion remains pending** until a workflow can coordinate
object storage, jobs and database history. No files or models are removed by
this endpoint.

## Verification

The existing schema supports these endpoints; no new migration or dependency
was needed. Run `make test-db` to exercise real PostgreSQL transactions in
disposable databases. Ten new project tests cover CRUD, pagination, two-account
isolation, input validation, batch rollback, concurrent additions, preserved
trained class maps, deletion conflicts and OpenAPI security.

Verified on 2026-10-20: all 31 API/database/configuration tests and 110 ML/setup
tests pass, along with lint.

Presigned uploads, upload completion, gallery reads and thumbnail processing are
now implemented in [UPLOADS.md](UPLOADS.md). Next: saving image annotations.
