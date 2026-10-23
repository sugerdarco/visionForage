# Model registry

The registry exposes models already registered by successful training workers.
Metadata browsing reads PostgreSQL. Evaluation reads use checksummed private
reports; download requests issue short-lived storage URLs. None of these routes
contacts the GPU host, loads ML models or reruns evaluation. No migration or
additional dependency is required.

## Routes

All routes require the owner's access bearer token and use `/api/v1`:

| Method and path | Response |
| --- | --- |
| `GET /projects/{project_id}/models` | Paginated model summaries, newest version first |
| `GET /models/{model_id}` | One model's metadata, frozen class map and training hyperparameters |
| `POST /models/{model_id}/promote` | Select this model as the project's preferred model; returns updated detail |
| `GET /models/{model_id}/evaluation` | Typed held-out metrics, ONNX parity and CPU latency |
| `GET /models/{model_id}/artifacts` | Names, sizes and SHA-256 values for the three downloadable files |
| `GET /models/{model_id}/artifacts/{artifact}` | A five-minute signed GET URL for `weights`, `onnx` or `metadata` |

Lists return `items`, `total`, `page` and `page_size`. Pages start at one; page size
defaults to 20 and is limited to 100. Model versions are unique within a project,
so ordering by descending version is deterministic. An empty registry or page
beyond the end returns an empty `items` array. As with other offset-paginated
lists, a newly registered model can shift subsequent pages.

Every summary includes:

- `id`, `project_id`, `job_id`, and the project's model `version`.
- `task_type` and the training job's `model_arch`.
- `dataset_version_id` and its numeric `dataset_version`.
- `input_size`, `is_promoted`, `created_at` and `updated_at`.

Detail adds `class_map` (class name to frozen output index) and `hyperparams`
(the job's stored effective parameters). The class map comes from the registered
model, never today's project class definitions. Adding a project class therefore
does not make an older model recognize it. The training configuration can contain
more requested epochs than were completed because of early stopping; it is not
a report of actual epoch count. Use the existing job metrics for epoch history.

The job detail/list response's successful `model_id` links directly to the detail
route. Jobs without a registered model do not appear in this registry. Models
are created by the worker after the verification gates, not through a browser
POST endpoint. `is_promoted` exposes the existing stored flag; a newly registered
model is not automatically promoted.

## Ownership and private artifacts

Foreign and nonexistent project/model IDs return 404. Missing, expired or refresh
tokens return 401. Invalid UUIDs or pagination parameters return 422. Successful
responses use `Cache-Control: no-store`.

Metadata list/detail queries select explicit public columns and join the associated
project, job and dataset version. They do not load the model's potentially large
`metrics` JSON or return `weights_key`, `onnx_key`, artifact indexes, signed URLs,
worker paths or raw logs. Separate evaluation/download routes resolve the stored
artifact index after checking ownership. Database foreign keys keep associated
records in the same project.

## Evaluation reports

The response is a discriminated union with `task_type` set to `classification` or
`detection`. Both variants include `model_id`, `split: "test"`, `checkpoint_epoch`,
the frozen `class_map`, `metrics`, `parity`, `latency` and
`latency_scope: "worker_cpu_model_only"`.

Classification adds `selection_val_loss`. Its metrics include loss, sample count,
accuracy, macro F1, ROC-AUC/method, positive class, per-class support/precision/
recall/F1/AUC, and a confusion matrix. Matrix rows are true classes and columns
are predicted classes, ordered by the frozen class map. Undefined AUC remains
`null`, not zero. The report's checkpoint epoch can be lower than the requested
training epochs because checkpoint selection uses validation results.

Detection adds `selection_split: "val"` and numeric `metric_settings`: AP confidence
floor and IoU thresholds, fixed precision/recall confidence and IoU, NMS IoU, and
the per-image detection cap. Metrics include `map50`, `map50_95`, fixed-threshold
micro counts/precision/recall and per-class AP/counts. AP is `null` for a class
without ground-truth boxes; mAP averages classes with ground truth and is `null`
when there are none. Support here counts objects, not images. Do not compare
fixed-threshold precision/recall with a differently configured evaluator.

Parity exposes the sample count, batches, maximum errors and tolerances, without
private image keys. The existing classifier 0.001 and detector 0.002-box/0.001-score
gates remain unchanged, with zero relative tolerance. Latency includes batch size,
warmup/measured counts and mean/p95 milliseconds. It measures ONNX Runtime on the
**worker host CPU**, excluding image decode, preprocessing, detection NMS, HTTP
and network time; it is not workstation or deployed-service latency.

The API reads `evaluation.json` and `onnx/verification.json`, at most 32 MiB each.
It checks their exact byte counts and SHA-256 against the registered artifact
index, checks manifest/checkpoint/ONNX hash linkage, validates typed metric shapes
and the stored parity result, and then constructs the public view. It does not
recompute predictions. Unknown/nested diagnostic fields, runtime identifiers,
loader audits and private image-key examples are omitted. Confident-mistake image
previews remain future work; no arbitrary image-storage keys are exposed here.

Missing, corrupted, mismatched or unsupported reports return a generic 409.
Legacy metadata-only models remain browsable even when a report cannot be read.
Storage outages return 503. No raw exception or private storage key is included
in these API error messages.

## Artifact downloads

The listing returns an array of `name`, `filename`, `content_type`, `bytes` and
`sha256`. It checks registered metadata but does not contact storage or issue
download permissions. A model without a complete valid downloadable index returns
409 instead of an invented or incomplete artifact list.

| Artifact name | Registered file | Intended use |
| --- | --- | --- |
| `weights` | `run/best.pt` | Retain the trained checkpoint |
| `onnx` | `onnx/model.onnx` | Use the exported model |
| `metadata` | `onnx/metadata.json` | Read preprocessing, class map and serving contract |

Request one artifact to get those fields plus `download_url` and `expires_in: 300`.
GET that URL directly, without the API bearer header. The signed request fixes an
attachment filename, content type and `private, no-store` cache control. Keep the
URL private: it contains a storage key and temporary access permission. It remains
usable until expiry even after local sign-out or a promotion change. Request a new
URL after expiry; do not edit its host, path or query parameters.

The API only resolves these fixed names. It validates the stored key as the model's
own `project/jobs/job/attempt-UUID/filename` and checks weight/ONNX references agree
with the model row. Clients cannot request another bucket, a raw object key, a
worker path or a log. Download issuance checks object existence and stored size;
it does **not** reread and hash a potentially 1-GiB binary. Verify the downloaded
bytes against the returned SHA-256. The existing create-only publication protects
normal writes, but a privileged storage administrator can still alter an object.

An unknown artifact name returns 404. Missing registered objects, changed sizes
or invalid indexes return 409; storage failures return 503. An object can still
become unavailable after a URL is issued. Raw training reports, manifests, audit
files and worker logs are not part of this download allowlist. Serving metadata
is the original registered JSON, while the evaluation endpoint is a filtered
public representation. Artifact SHA-256 values describe the original stored bytes,
not a filtered evaluation response.

## Promotion

POST promotion takes no body. It requires a model whose training job succeeded
and returns 409 otherwise. The API locks the owning project, clears its previous
preferred model and selects the requested model in one transaction. It shares
the project's registration lock and keeps the existing database constraint of
at most one promoted model per project. Different projects remain independent.

Promoting the already preferred model is safe to retry and leaves its timestamps
unchanged. Concurrent requests serialize; the last serialized promotion determines
the final default. Each response describes the state at its transaction, which a
later request can change. The API does not rank candidates or choose a model
using held-out test results. Promotion does not deploy, restart or change any
inference endpoint, and no worker/storage access is needed.

## Checks

`make test-db` checks both routes against disposable PostgreSQL databases. Registry
coverage includes empty results, version ordering, pagination, cross-owner and
same-owner cross-project isolation, access-token requirements, classification and
detection metadata, frozen class maps, job/model linking, private-field exclusion
and generated OpenAPI schemas. These are metadata fixtures, not new GPU runs.
Promotion checks include concurrent requests, retry behavior and cross-project
isolation. `make test-uploads` also checks real private MinIO downloads, response
headers, SHA-256 matching, unsigned/tampered URL rejection, report provenance,
redaction, read-size bounds, missing objects and storage outages. Fixture binary
files test transport; they are not trained models. The report reader was also
validated locally against the retained real detector and two classifier bundles
from the queued-worker verification, without retraining them.
