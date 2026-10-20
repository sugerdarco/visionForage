# Frozen dataset versions

Dataset versions connect live project labels to reproducible training inputs.
Creating a version freezes the current labeled images, class map and train/val/test
assignments into the schema-1 manifest already consumed by `vf_ml.manifest`.
The API also writes an audit document with original-file and decoded-pixel hashes.
Both documents live in private MinIO storage; PostgreSQL stores their keys,
checksums, split settings and counts in an immutable dataset-version row.

This first builder runs synchronously on the local API's CPU. It requires neither
the GPU host nor a Celery worker, and does not import PyTorch, timm or Ultralytics.
One build per API process limits local resource use. Moving larger builds to a
durable CPU queue remains an extension; the current limits make the request and
failure behavior concrete without introducing another job lifecycle yet.

## API

All routes require an access bearer token and the project's owner. Paths below
are relative to `/api/v1`:

| Method | Path | Behavior |
| --- | --- | --- |
| POST | `/projects/{project_id}/dataset-versions` | Build and register a new frozen version; 201 |
| GET | `/projects/{project_id}/dataset-versions` | Paginated list, newest version number first |
| GET | `/projects/{project_id}/dataset-versions/{version_id}` | Read one version and obtain fresh private download URLs |

Creation accepts `{}` for the defaults, or:

```json
{
  "seed": 42,
  "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}
}
```

The seed is an integer from zero through `2**32 - 1`. All three ratios must be
positive finite numbers below one and sum to one within `1e-9`. Clients cannot
choose object keys, class indices, version numbers or image IDs. The builder
selects every currently `labeled` image; `unlabeled` and `in_review` rows are
excluded and counted separately. Labeled detection images with no boxes are
included as reviewed backgrounds. Classification requires exactly one tag.
Labels must be human-reviewed, with classes belonging to this project.

Responses include `id`, `project_id`, `version`, `manifest_key`, `split_ratios`,
`counts`, `created_at`, `manifest_url`, `audit_url` and `urls_expire_in: 300`.
`counts` includes:

- `images`, `labels` and `pixel_groups`, each broken down by train/val/test.
- `classes`: per-split label counts by frozen class name; for detection these
  count boxes, not images containing that class.
- `excluded`, `duplicate_groups`, `class_map`, `seed` and `split_method`.
- `warnings`: validation/test splits missing any class.
- `manifest_sha256`, `audit_sha256` and `audit_key`.

GET list returns `items`, `total`, `page` and `page_size`. Page defaults to one;
page size defaults to 20 and is limited to 100. All API responses are `no-store`.
Download URLs expire after five minutes; fetch the version again for fresh URLs.
Unsigned object reads are denied. `S3_DATASETS_BUCKET` defaults to `vf-datasets`,
which the existing local storage bootstrap already creates.

Every successful POST creates a new version, even if its manifest matches an
earlier version. It is **not an idempotent endpoint**. After a lost response,
inspect the version list before submitting another creation request. Version
numbers increase within a project and are allocated under the project row lock.

## Snapshot and split rules

The API first locks the project and its image rows briefly, copies the class map,
statuses and labels into memory, then releases those locks before reading storage.
The copy is the snapshot boundary. Label edits or uploads completed afterward
belong to a future version. Builds do not block editing during image decoding.
If several API processes build concurrently, final registration takes the project
lock again so each receives a distinct version number.

Every selected original is downloaded with a bounded read. Its bytes must match
the registered SHA-256 and its dimensions must match the image row. The same
JPEG/PNG, mode and orientation rules used by uploads apply here. The decoded image
is converted to RGB, then hashed together with its dimensions. Images with exactly
the same RGB pixels form one group, even when their compressed bytes differ.

Groups are ranked by `sha256(seed:pixel_sha256)`. Classification ranks groups
separately within each class. It needs at least three unique pixel groups per
class. Identical classification pixels with conflicting classes are rejected.
Detection ranks all groups together and needs at least three groups overall;
it does not perform multi-label stratification. Duplicate detection images keep
their original boxes, including differing annotations, but always share a split.

For each ranked set, the builder rounds the requested test group count, then the
validation group count, reserving at least one group for each split. Remaining
groups train. Ratios therefore target **pixel-group counts**, not exact image
counts. Small classes or uneven duplicate groups can make realized image ratios
differ substantially; the response reports actual counts.

Every project class must have at least one training example. If it does not,
creation fails with 422. Detection classes missing from validation or test produce
explicit warnings and zero counts. Add labeled examples or choose a seed based
on dataset coverage, not on model performance against the held-out test set.

The same copied inputs, decoder version, seed and ratios yield the same manifest
bytes. Adding images or changing labels can change group ranks and move existing
images between splits in a **new** version. Keep one frozen version when comparing
models; do not treat evaluation scores from changed splits as directly comparable.
Import source splits are not restored by this builder.

Grouping prevents exact-pixel leakage. It does not detect near duplicates,
augmentations or related frames, and the application does not yet store acquisition
or parent-image IDs. A group can contain several samples, so duplicate examples
still affect metrics within their assigned split. These limitations also apply
when designing the later active-learning comparison.

## Artifacts and immutability

The manifest keeps the existing strict training contract:

```json
{
  "schema_version": 1,
  "task_type": "classification",
  "class_map": {"ok": 0, "defect": 1},
  "samples": [
    {
      "image_key": "<project>/originals/<sha256>.png",
      "split": "train",
      "labels": [{"class_index": 0, "bbox": null}]
    }
  ]
}
```

The abbreviated example shows one sample. Actual versions contain nonempty
train/val/test splits. Detection labels use normalized top-left boxes. Image keys
refer to `vf-images`; the worker will download those objects into its local data
directory while preserving keys, then use the unchanged manifest loader.

Artifacts use a unique attempt directory:

```text
vf-datasets/{project_id}/v{version}/{attempt_uuid}/manifest.json
vf-datasets/{project_id}/v{version}/{attempt_uuid}/audit.json
```

The extra attempt UUID extends the plan's original `v{version}/manifest.json`
layout so a failed build's orphan objects cannot collide with a later attempt.
Both writes use `If-None-Match: *`; existing keys cannot be overwritten by this
write path. The local MinIO integration tests verify its 412 response on an
overwrite attempt. This follows [S3 conditional-write semantics](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).

The database's existing trigger rejects updates to dataset-version rows. There
are no edit/delete version endpoints. Later label changes, class additions and
new uploads leave the version's class map and labels unchanged. The audit records
the seed, split algorithm, decoder version, per-image IDs, object keys, dimensions,
original hashes, pixel hashes and assigned splits. Its labels live in the paired
manifest, whose checksum the audit includes.

Original images are referenced, not copied per version. This saves storage and
uses the application's existing content-addressed original keys. Immutability
here is enforced by application behavior and conditional artifact writes; it is
not administrator-proof object locking. A privileged storage operator can still
change or delete objects. The future training download step must verify the
artifact and original checksums before using them. General original deletion
must respect dataset-version references.

## Limits and failures

| Build limit | Value |
| --- | --- |
| Project images, including excluded statuses | 10,000 |
| Project classes | 1,000 |
| Labels in selected images | 100,000 |
| Original bytes read | 1 GiB total; 20 MiB per image |
| Decoded pixels | 1 billion total; existing 20-megapixel per-image limit |
| Manifest or audit document | 32 MiB each |
| Concurrent builds | One per API process |

An overlapping build returns 503 with `Retry-After: 5`. Invalid labels, images,
limits or insufficient split coverage return 422. Missing original objects return
409. Storage/database outages return generic 503 responses. Ownership failures
return 404 without disclosing another account's data.

The row commits only after both artifact writes succeed. A failed object write or
database commit can leave unreferenced artifacts because PostgreSQL and MinIO do
not share a transaction. A retry uses a fresh attempt UUID and never overwrites
those objects. General orphan cleanup remains pending. No migration or package
update was needed: the existing immutable table and JSON `counts` column hold
the snapshot metadata.

## Verification

`make test-db` includes split and request-validation tests. `make test-uploads`
includes real PostgreSQL/MinIO checks for private manifests, conditional writes,
ownership, concurrent numbering, edits during decoding, immutable history,
failure rollback, metadata limits and loading a generated manifest with the
existing `vf_ml.manifest` module. The API remains free of ML runtime imports.

Validated for this milestone: 57 API/database/parser tests, 34 storage/worker
integration tests and 110 ML/setup regression tests passed, together with lint.
