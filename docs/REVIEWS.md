# Reviewing predictions

The backend now closes the prediction-to-dataset loop. A human submits corrected
labels; acceptance copies the prediction image into the project, saves human
annotations and records a receipt. Retraining uses a newly created dataset version.
Existing frozen versions and trained models remain unchanged.

All routes require the owner's access JWT and return private, non-cacheable data.
Unknown or foreign IDs return 404. A project being deleted returns 409.

| Route | Contract |
| --- | --- |
| `GET /api/v1/projects/{id}/review-queue?page=1&page_size=20` | Pending low-confidence predictions, lowest confidence first, then creation time and ID. Returns `items`, `total`, `page`, `page_size`; maximum size 100. |
| `POST /api/v1/review/{prediction_id}/accept` | Save corrected annotations and return `prediction_id`, `image_id`, `reviewed_at`. |

Queue items include model/deployment IDs, model version, original output,
confidence, latency, timestamp and a private image URL valid for five minutes.
Fetch the page again when a URL expires. Offset pagination can shift as users
accept predictions or new predictions arrive.

Classification acceptance body:

```json
{"annotations": [{"class_id": "PROJECT_CLASS_UUID"}]}
```

Detection uses the same body with normalized top-left `bbox: [x, y, width, height]`
on each annotation. An empty array is a reviewed negative detection image.
Classification requires exactly one class and no box. Classes must belong to the
project; boxes must fit the image. Validation matches [annotation saves](ANNOTATIONS.md).
The server never treats a model guess as a human correction automatically.

Acceptance locks the project and prediction and commits the labels and receipt
in one database transaction. Repeating the same labels returns the same receipt;
a different correction for an already accepted prediction returns 409. Annotation
order does not affect retry matching. Project images are deduplicated by SHA-256.
An existing unlabeled image can be labeled; matching human labels can be reused.
Conflicting existing annotations return 409 rather than overwriting them.

Accepted predictions disappear from the review queue. History exposes
`review_image_id` and `reviewed_at`; `sent_to_review` stays true so historical
low-confidence statistics do not change. Missing stored images return 409;
storage outages leave acceptance pending for retry. Thumbnails recover through
the existing CPU worker.

Create a new frozen version after corrections, then submit training against it.
The backend workflow is complete; the browser review screen and the planned
active-learning comparison experiment are separate deliverables.

`make test-uploads` covers concurrent acceptance, retries, ownership, storage
failure, classification/detection validation, duplicate-image conflicts and
inclusion in the next frozen version without changing an earlier version.
