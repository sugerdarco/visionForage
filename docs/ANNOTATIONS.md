# Image annotations

The labeling API connects uploaded images and project classes to the live labels
that a future dataset-version endpoint will freeze into a training manifest.
It runs locally in FastAPI and writes PostgreSQL. Saving labels does not fetch
image bytes, enqueue work, or use the GPU host.

Both routes require an access bearer token and the image's project owner:

| Method | Path under `/api/v1` | Result |
| --- | --- | --- |
| GET | `/images/{image_id}/annotations` | Current image status and annotations |
| PUT | `/images/{image_id}/annotations` | Replace the complete annotation set |

Foreign and nonexistent image IDs both return 404. Successful reads and saves
return 200 with `Cache-Control: no-store`. Invalid input returns 422 without
changing existing labels.

## Classification

Send the UUID from the project's class list:

```json
{
  "annotations": [
    {"class_id": "11111111-1111-4111-8111-111111111111"}
  ]
}
```

A labeled classification image has exactly one class and no bounding box.
Omitting `bbox` or sending `null` has the same meaning. Multiple classes are
rejected because the classifier learns one mutually exclusive target per image.
Class IDs must belong to this image's project, even if the same account owns
another project with a class of the same name.

## Detection

```json
{
  "annotations": [
    {
      "class_id": "11111111-1111-4111-8111-111111111111",
      "bbox": [0.1, 0.2, 0.3, 0.4]
    }
  ]
}
```

Boxes are `[x, y, width, height]`, normalized to image dimensions. `x` and `y`
are the **top-left corner**, not YOLO's center coordinates. For a 200 × 100 image,
this example starts at pixel `(20, 20)` and has size `60 × 40`. The future YOLO
import/export adapters must convert between these conventions.

Coordinates must be finite JSON numbers; strings and booleans are rejected.
Origins are in `[0, 1]`, extents in `(0, 1]`, and the box must fit inside the
image. Edge sums allow the database's existing `1e-9` floating-point tolerance.
Each detection annotation requires a box. Multiple boxes may use the same class;
exact duplicate boxes are preserved rather than silently discarded. Requests
are limited to 1,000 annotations to bound each replacement transaction.

## Empty images and review

With no explicit `status`, a nonempty list becomes `labeled` and an empty list
becomes `unlabeled`. To clear labels and return an image to the labeling queue:

```json
{"annotations": []}
```

For a detection image that a person inspected and found no objects, explicitly
mark the empty result as labeled:

```json
{"annotations": [], "status": "labeled"}
```

That distinction lets future version creation include reviewed negative images
while excluding images nobody has labeled. Classification requires a class even
for an acceptable image, so this empty labeled form is rejected there.
`status: "unlabeled"` is accepted only with an empty list. The save endpoint
cannot set `in_review`; review ingestion remains separate future work.

Every saved annotation has `source: "human"` and `confidence: null`. Clients
cannot submit those fields, annotation IDs or image IDs in the body. Saving
reviewed model suggestions replaces their provenance with human labels and
moves the image out of `in_review`. The model-review ingestion workflow itself
is not implemented yet.

## Responses, retries and concurrent edits

GET and PUT share this response shape:

```json
{
  "image_id": "22222222-2222-4222-8222-222222222222",
  "status": "labeled",
  "annotations": [
    {
      "id": "33333333-3333-4333-8333-333333333333",
      "class_id": "11111111-1111-4111-8111-111111111111",
      "bbox": null,
      "source": "human",
      "confidence": null
    }
  ]
}
```

Rows are returned in UUID order; request order has no semantic meaning.
Repeating an identical human-label set, including in a different order, preserves
annotation IDs and timestamps. A changed set replaces all its rows, so clients
must use the returned IDs. Clearing an already empty set is also safe to retry.

PUT locks the image row, validates all labels, then replaces labels and updates
status in one transaction. Different images can be edited independently.
Concurrent saves on one image serialize; the last save wins as a complete set,
with no merged or partially deleted boxes. GET uses one database statement so
status and annotations come from the same committed snapshot.

Whole-set replacement is simpler for a labeling canvas than separate requests
for every drawn, moved or deleted box. The tradeoff is more writes when one box
changes. There is no revision/ETag conflict detection yet: a stale browser can
overwrite a newer save. Clients should serialize autosaves per image and avoid
retrying an old request after a newer edit has succeeded.

Gallery status and class filters immediately reflect saved labels. Existing
dataset versions and model class maps are not edited. Training will use frozen
manifests, not these live annotation rows.

## Verification

Run `make test-db` against local PostgreSQL. Eleven new annotation tests use
disposable databases and cover ownership, access tokens, task rules, cross-project
class rejection, malformed/non-finite boxes, rollback on invalid batches,
idempotent saves, reviewed negatives, model suggestions, gallery filters,
concurrent replacements and the OpenAPI contract.

Validation for this milestone: all 42 API/database tests, 10 upload/worker
integration tests and 110 ML/setup regression tests passed, together with lint.

The existing schema and dependencies support this milestone; no migration or
package update is needed. Dataset imports and version creation come next.
