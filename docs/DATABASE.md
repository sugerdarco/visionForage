# Application database

The first migration creates the build plan's 11 metadata tables in PostgreSQL.
The Python models live in `services/api/vf_api/models.py`; they have no dependency
on the training package. Images, manifests and model binaries remain in MinIO;
database rows store their object keys.

| Tables | Connection to the workflow |
| --- | --- |
| `users`, `projects`, `classes` | A user owns projects; each project defines its model class indices. |
| `images`, `annotations` | Images belong to a project; annotations reference an image and a class. |
| `dataset_versions` | Each project gets numbered snapshots pointing to frozen manifest files. |
| `training_jobs`, `training_metrics` | A job references a version; epoch metrics belong to the job. |
| `models`, `deployments` | A job produces a versioned model; deployments reference that model. |
| `predictions` | Each logged result references a deployment and its stored input image. |

The API opens a session for a unit of work, validates project ownership and
commits account/project/class/image transactions. Workers use the same job and
metric records, but training still reads a frozen manifest rather than querying
live image and annotation tables. [Account HTTP routes](API.md) and
[project/class routes](PROJECTS.md) are implemented. [Uploads and CPU thumbnails](UPLOADS.md)
now use image rows; [annotation saves](ANNOTATIONS.md) atomically replace labels
and update image status. [Dataset versions](DATASET_VERSIONS.md) now store frozen
manifest keys, split settings and checksum/count metadata. The [GPU worker](GPU_WORKER.md)
claims training jobs, saves metrics and registers verified models. Heartbeats and
ownership tokens prevent expired workers from publishing results.

## Setup and migration

Start the local services using [LOCAL.md](LOCAL.md), then run from the repo root:

```bash
make api-env UV="$HOME/.local/share/visionforge/tools/bin/uv"
make test-db
make migrate
make db-current
```

`api-env` creates a dedicated Python 3.11.16 `.venv` and synchronizes the pinned
`requirements/api.txt`. It does not modify the existing CPU/GPU training
environments. Reserve `.venv` for the API: dependency synchronization removes
packages that are not in that list. Direct dependencies and their purposes are
listed in `requirements/api.in`.

Settings read the existing `.env` and environment variables. The latter take
precedence. `POSTGRES_HOST` defaults to `127.0.0.1`; a future Compose API container
will use `postgres` and the container port 5432. Existing `.env` files need no
change for host-side migration commands. Passwords are represented as secrets,
and the database URL is constructed with SQLAlchemy's `URL.create` so special
characters do not break parsing. No URL or password is stored in `alembic.ini`.

`make migrate` applies revisions through `0003`. Revision `0002` adds training
dispatch and cancellation timestamps; `0003` adds heartbeat and ownership fields
and updates recovery/dispatch indexes. See [training jobs](TRAINING_JOBS.md) and the
[worker upgrade sequence](GPU_WORKER.md#updating-existing-installations).
Running it again leaves an up-to-date
database unchanged. It creates the application tables plus Alembic's revision
table; it does not seed accounts, import datasets or start an HTTP server.

## Choices and constraints

- **SQLAlchemy 2 + psycopg:** typed models and explicit transactions without a
  separate persistence framework. The synchronous driver is straightforward to
  reuse in workers; future HTTP handlers must run blocking database work in
  synchronous handlers or an appropriate thread pool.
- **Alembic:** schema changes are explicit, reviewed revisions. The migration
  contains its own table definitions and never imports current models to create
  tables, so later model edits cannot silently change an old migration.
- **UUIDs and timezone-aware timestamps:** PostgreSQL generates IDs and creation
  times for both ORM and SQL inserts. Database triggers maintain `updated_at` on
  mutable records even when a worker uses raw SQL.
- **JSONB:** hyperparameters, metric summaries, class maps, split ratios and
  counts can vary by model. Their top-level object shape is checked; detailed
  payload validation belongs to the service layer.
- **Status strings with CHECK constraints:** valid values are enforced without
  introducing PostgreSQL enum types that need separate migration handling.

Class names and class indices are unique within their project. Image
SHA-256 deduplication is also project-scoped, so another project may use the same
image bytes. Dataset and model version numbers must be positive and unique per
project. Composite foreign keys prevent jobs from using another project's
dataset version and models from referencing another project's job. There is
at most one model per job and one promoted model per project.

Boxes use a four-element double-precision array `[x, y, width, height]`. Coordinates
are normalized, sizes must be positive and boxes must fit inside the image;
the edge check uses the existing manifest loader's `1e-9` rounding allowance.
The bind type converts integer and float coordinates to homogeneous floats for
psycopg. Null boxes are supported for classification. Confidence values must be
between zero and one; epoch metrics and latency reject non-finite numbers.

Dataset-version rows reject UPDATE statements: changes require a new version.
The version-creation service also uses conditional create-only MinIO writes and
a unique artifact directory per attempt; it never overwrites an existing manifest. Foreign keys do not
cascade-delete history, so project removal will need an explicit cleanup workflow.

This schema is not an authorization layer. Project routes enforce ownership and
assign append-only class indices. Annotation routes enforce same-project
image/class pairing and task-specific label rules. Training routes and the worker
enforce allowed job transitions and successful-job requirements for registration.
Upcoming registry and deployment routes must preserve these boundaries. The
`anomaly` task value is reserved by the plan; no anomaly training is implemented.

## Validation and future changes

`make test-db` creates a uniquely named disposable PostgreSQL database using the
configured server and credentials. The local development database user needs
CREATEDB permission. Tests apply the migration, exercise a complete record chain,
reject invalid links/values and duplicates, check triggers, downgrade to empty,
upgrade again and compare the resulting schema with the models. The disposable
database is removed afterward; application tables are not rolled back by tests.
Configuration tests also check password redaction and the absence of ML imports.

After changing models, generate and review a **new** revision:

```bash
.venv/bin/python -m alembic -c services/api/alembic.ini revision --autogenerate -m "Describe the schema change"
make test-db
make migrate
.venv/bin/python -m alembic -c services/api/alembic.ini check
```

Do not edit a migration after it has been applied. Autogeneration does not cover
every database object: review CHECK constraints and trigger changes explicitly.
The current integration tests target the initial schema and must evolve with
later revisions. See the [Alembic autogeneration documentation](https://alembic.sqlalchemy.org/en/latest/autogenerate.html).

Revision `0003` validation passes 66 database/API tests and 64 service integration
tests against local PostgreSQL, Redis and MinIO, along with lint. It is applied
locally, and Alembic's schema comparison reports no pending changes. The tests
include full downgrade/upgrade coverage, worker heartbeat recovery, ownership
checks and lost queue messages.
