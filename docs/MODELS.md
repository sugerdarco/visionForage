# Model registry

The registry exposes models already registered by successful training workers.
The API reads PostgreSQL metadata; browsing does not contact the GPU host, load
weights, sign storage URLs or rerun evaluation. No migration is required.

## Routes

Both routes require the owner's access bearer token and use `/api/v1`:

| Method and path | Response |
| --- | --- |
| `GET /projects/{project_id}/models` | Paginated model summaries, newest version first |
| `GET /models/{model_id}` | One model's metadata, frozen class map and training hyperparameters |

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

Queries select explicit public columns and join the associated project, job and
dataset version. They do not load the model's potentially large `metrics` JSON
or return `weights_key`, `onnx_key`, artifact indexes, signed URLs, worker paths
or raw logs. This keeps the list small and leaves storage authorization to a
dedicated future download endpoint. Database foreign keys keep these records in
the same project; HTTP ownership checks restrict reads to its owner.

Final evaluation metrics, parity and latency summaries, authorized artifact
downloads and model promotion are the next registry increments. These endpoints
do not yet expose those reports, even though the worker stores them privately.
Promotion and deployment are separate operations; neither is triggered by reads.

## Checks

`make test-db` checks both routes against disposable PostgreSQL databases. Registry
coverage includes empty results, version ordering, pagination, cross-owner and
same-owner cross-project isolation, access-token requirements, classification and
detection metadata, frozen class maps, job/model linking, private-field exclusion
and generated OpenAPI schemas. These are metadata fixtures, not new GPU runs.
