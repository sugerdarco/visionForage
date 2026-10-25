# VisionForge walkthrough

Start the local stack with `make up`, then open `http://127.0.0.1:3000`.
The remote GPU worker must be running to consume training jobs; see
[worker setup](GPU_WORKER.md). Its connection details belong in private local
configuration. No GPU or sudo access is required by the browser/API containers.

1. Create a classification project with `ok` and `defective` classes. Import a
   labeled ImageFolder ZIP, inspect the images, then freeze a dataset version.
2. Select **Train on this version**. Check the selected version and submit a run.
   The training page streams progress; registration follows evaluation and ONNX
   verification. Models provides test reports, private downloads and promotion.
3. Select **Deploy this model**, set a review threshold and create the endpoint.
   Save its one-time key. Upload a separate image under **Test prediction**.
4. Open **Monitoring** to see the recorded request, latency and confidence. Open
   **Review**, correct the label and explicitly accept it as a human annotation.
5. Freeze a new version and train on it. The earlier frozen version is unchanged.
   Deploy the new model, verify a prediction, then stop the old endpoint.

The optional browser scenario automates this workflow using 240 public casting
images, two epochs per job and one separate defective image. It asserts the actual
job's dataset ID and the deployed model ID, so a cached form cannot silently train
or deploy an older selection. After review, the next training split contains 169
images instead of 168; validation and test each contain 36. These short runs test
integration and are not the full model-quality benchmarks in [RESULTS.md](RESULTS.md).

Run the standard browser suite without a GPU worker consuming its queue. For the
live scenario, install Playwright's video helper (`npx playwright install ffmpeg`
from `apps/web`), start an allocated worker, and run:

```bash
E2E_REAL_TRAINING=1 \
E2E_CASTING_ARCHIVE=/path/to/casting-240.zip \
E2E_CASTING_IMAGE=/path/to/separate-defective.jpeg \
npm run test:e2e --prefix apps/web -- live-training.spec.ts
```

The input image must have a known defective label and be absent from the ZIP.
Use `E2E_KEEP_DEMO=1` to retain the project for inspection; both endpoints are
stopped after verification. The test stores local access details and results under
ignored `artifacts/`, and video under ignored `apps/web/test-results/`. Neither
credentials nor raw browser traces should be committed. The portable verification
summary is [browser-workflow-20261025.json](results/browser-workflow-20261025.json).

Separately, [notebook 03](../notebooks/03_active_learning.ipynb) reports the full
10%-seed active-learning experiment. Uncertainty and random acquisition tied at
99.7203% test accuracy; the demo makes no claim of improved model quality.
