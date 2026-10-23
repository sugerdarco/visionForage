# Local ONNX inference service

The inference service loads one registered model bundle and serves CPU predictions
through FastAPI. It is separate from the control API and GPU worker. Run it manually
as described below or use the [Docker deployment API](DEPLOYMENTS.md) for automatic
bundle download, create/status/stop and stable Traefik routes. Deployed models also
support [durable prediction logging and monitoring](MONITORING.md).

The request path is: authenticated multipart upload → bounded image validation →
saved preprocessing → ONNX Runtime → classification probabilities or detector
filtering/NMS → JSON response. No training checkpoint is loaded or executed.

## Environment and startup

From the repository root, with the existing uv tool installed:

```bash
make inference-env UV="$HOME/.local/share/visionforge/tools/bin/uv"
make test-inference
cp services/inference/.env.example services/inference/.env
chmod 600 services/inference/.env
```

Fill the private environment file before starting. Obtain the `onnx` and `metadata`
downloads and their SHA-256 values from the [model registry](MODELS.md). Place the
files together as `model.onnx` and `metadata.json` in a private bundle directory.
Use the registered model ID and version, not its training job ID. The standalone
service trusts this operator-supplied identity; the deployment API binds it to the
registry automatically. It has no database or storage credentials.

Generate a random endpoint key locally and retain it privately. For example:

```bash
python3 - <<'PY'
import hashlib, secrets
key = secrets.token_urlsafe(32)
print("Private API key:", key)
print("INFERENCE_API_KEY_SHA256=" + hashlib.sha256(key.encode()).hexdigest())
PY
```

Put only the digest in `INFERENCE_API_KEY_SHA256`; clients use the original key.
Do not commit either value. The remaining configuration is:

| Variable | Value |
| --- | --- |
| `INFERENCE_BUNDLE_DIR` | Directory containing the two downloaded files |
| `INFERENCE_MODEL_ID` | Registered model UUID |
| `INFERENCE_MODEL_VERSION` | Positive project model version |
| `INFERENCE_SLUG` | Lowercase letter followed by lowercase letters/digits/hyphens; up to 63 characters |
| `INFERENCE_MODEL_SHA256` | Registry digest for `model.onnx` |
| `INFERENCE_METADATA_SHA256` | Registry digest for `metadata.json` |
| `INFERENCE_CPU_THREADS` | 1–8; default 2 |

```bash
make inference
```

This starts one Uvicorn process on `127.0.0.1:8010`, loading
`services/inference/.env`. Override its path with `INFERENCE_ENV_FILE` if needed.
Do not reuse the control API's private environment file. `/healthz` is process
liveness, `/readyz` confirms bundle initialization, and `/docs` describes the
prediction contract. Invalid checksums, metadata or graph contracts prevent
startup. Stop with Ctrl+C. Key rotation currently requires changing the digest
and restarting the service.

Bundle metadata is capped at 1 MiB and the graph at 1 GiB. Both checksums are
verified against configuration; metadata must also identify that same graph.
Only schema 1, opset-17 metadata, supported task/preprocessing conventions and
matching float32 NCHW input/output shapes are accepted. The graph is loaded from
verified bytes so external weight files cannot be resolved from host paths.
Keep the bundle directory read-only during operation; this is not a model upload
endpoint or a general-purpose graph execution service.

## Prediction contract

`POST /predict/{slug}` requires `X-API-Key` and exactly one multipart file named
`image`. The key is checked before reading the body. Send JPEG or PNG with the
matching part content type. Example after setting private shell variables:

```bash
curl --fail-with-body \
  -H "X-API-Key: $INFERENCE_API_KEY" \
  -F "image=@sample.png;type=image/png" \
  "http://127.0.0.1:8010/predict/$INFERENCE_SLUG"
```

Both response types include `request_id`, `model_id`, `model_version` and
`latency_ms`. Classification adds `label`, `confidence` and a `scores` mapping in
frozen class order. Detection adds `boxes`, each with `label`, `confidence` and
normalized **top-left** `x`, `y`, `w`, `h` in the original image coordinate system.
An empty `boxes` array is a valid result. Deployed services also queue the image
and output for durable collection.

Standalone `request_id` is transient. Deployed services also return a matching
`prediction_id` after durably queuing the image and result for collection into
MinIO/PostgreSQL; see [logging behavior](MONITORING.md). The measured
latency includes decoding, preprocessing, model execution and postprocessing,
starting after upload parsing. It excludes upload/network time and is distinct
from the registry's model-only ONNX benchmark.

| Condition | HTTP status |
| --- | --- |
| Missing/incorrect API key | 401 |
| Wrong slug, after valid authentication | 404 |
| Upload exceeds limits | 413 |
| Wrong request or image part media type | 415 |
| Malformed multipart, extra fields/files, invalid image or encoding mismatch | 422 |
| Another prediction/upload is active | 429, `Retry-After: 1` |
| Upload takes longer than 30 seconds | 408 |
| Model execution fails | 503 with a generic error |

Images must be RGB or grayscale, single-frame JPEG/PNG, at least 10×10 and at most
20 megapixels, with normalized EXIF orientation. Transparency, palette images
and animation are rejected. File bytes are capped at 20 MiB; the entire multipart
body is capped at that limit plus 64 KiB of overhead, even without Content-Length.
Temporary multipart files are closed on success, parsing failure and disconnect.
These rules match the control API's accepted image policy.

One request per process holds capacity from upload through postprocessing; others
receive 429 instead of an unbounded inference backlog. The model call runs off the
ASGI event loop, allowing health checks while it executes. Run one server process
per model for this increment. Runtime exceptions do not expose paths or private
diagnostics in HTTP responses. Responses use `Cache-Control: no-store`.

## Preprocessing and runtime choices

Classification calls the same `vf_ml.transforms.classification_transform` used
by training/evaluation, with the exported resize/interpolation/mean/std settings.
It resizes the full image without a center crop, then applies softmax to ONNX
logits. The isolated serving environment therefore includes **CPU-only PyTorch
and Torchvision for preprocessing**, while the model itself runs in ONNX Runtime.
It does not install timm, Ultralytics, CUDA, Celery or database/storage clients.

This is a deliberate size/correctness tradeoff. A local probe found that replacing
the saved tensor bicubic resize with Pillow changed some uint8 pixels, including
on 300×300 inputs. Sharing the exact transform avoids silently changing predictions.
A smaller implementation can replace it only after equivalent preprocessing is
verified. See [Torchvision's resize contract](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html)
for its interpolation and antialias options.

Detection uses headless OpenCV with the pinned core version, reproducing the
exported Ultralytics 8.3.221 loader: resize the long side, ceil the other dimension,
center-pad with 114, convert BGR to RGB/CHW, and divide float32 pixels by 255.
Padding and actual per-axis resize ratios are retained for mapping boxes back.
Multi-label, class-aware NumPy NMS uses the saved confidence/IoU settings, at most
30,000 sorted candidates and the saved detection limit (up to 300). Boxes are
clipped to the original image; zero-area clipped results are omitted. Exact ties
use stable input order and need not match every alternative NMS implementation.

Detection keeps `CPUExecutionProvider` with `ORT_DISABLE_ALL`, as required by its
verified bundle. Classification uses its export-time default graph optimizations.
Both use sequential graph execution, the configured intra-op thread count and
one inter-op thread. These controls follow the [ONNX Runtime session API](https://onnxruntime.ai/docs/api/python/api_summary.html).
They do not change the approved raw-output parity tolerances.

## Verification and next work

`make test-inference` exercises a small real ONNX graph for HTTP/runtime behavior,
plus geometric/NMS cases. It covers authentication, media/size limits, shape/hash
failures, busy responses and recovery, invalid image modes, redacted errors and
secret-safe settings errors. The embedded graph is a 263-byte channel-mean test
fixture, not a trained model; no pretrained weights or dataset download is needed.

Additional local checks used the retained real queued-worker bundles:

- 32 classifier verification images: HTTP probabilities matched the reference
  within absolute `1e-7`, with the same winning classes.
- 32 detector verification images: preprocessing was identical to the pinned
  loader; NMS count/classes matched, with normalized boxes/scores within `1e-6`.
- Five additional synthetic aspect ratios matched detector preprocessing exactly.
- Both real bundles ran over loopback HTTP in the isolated serving environment;
  temporary servers were stopped afterward. Private logs and binaries stay outside Git.

These are serving compatibility checks, not new quality or latency benchmarks.
[Docker packaging and deployment create/status/stop](DEPLOYMENTS.md) now include
automatic verified bundle preparation, one-time endpoint keys and Traefik routing
with browser CORS support, prediction logging and monitoring APIs. The browser
playground remains pending. The standalone service itself has no CORS middleware; deployed browser
clients use the gateway described in the deployment guide.
