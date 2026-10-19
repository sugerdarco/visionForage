# Baseline results

## Casting classifier — casting-baseline-20261019

This is the first full-data casting baseline, trained on 2026-10-19. It uses
pretrained ResNet-18, the frozen casting-v2 manifest, and the existing training
recipe. Hyperparameters were fixed before test evaluation. The checkpoint was
selected solely by validation loss; classification uses argmax with no test-tuned
threshold. NEU-DET baseline quality results and active learning remain pending.

The [machine-readable snapshot](results/casting-baseline-20261019.json) contains the actual
history, validation/test reports, ONNX metadata, verification, and artifact hashes.
[Notebook 01](../notebooks/01_casting_baseline.ipynb) reads that snapshot and plots
the learning curves and confusion matrix without needing model weights or a GPU.
Training source: [efd78de](https://github.com/sugerdarco/visionForage/commit/efd78de5aa2d8ffcaebb560a5f2bfc606d777c21).

### Dataset and training

| Split | OK | Defective | Total |
| --- | ---: | ---: | ---: |
| Train | 2,389 | 3,194 | 5,583 |
| Validation | 422 | 564 | 986 |
| Test | 262 | 453 | 715 |

The supplied test split is preserved. Preparation removes 64 exact training copies
of test images and splits the remaining training images reproducibly by class.
Manifest SHA-256: `fbce28e2e462ad9ccd0e046a4cbbef1dca7535ea710efa1eccfc0bf4f7bc7865`.
Image bytes were checked against the preparation audit before this run; local
validation/test copies were checked before CPU evaluation.

The dataset already contains augmented images without parent-image identifiers.
Exact-pixel deduplication cannot eliminate every possible augmented relative
across splits. These are results on this public dataset split, not evidence of
independent production or cross-factory generalization.

| Setting | Value |
| --- | --- |
| Backbone | `resnet18.a1_in1k`, all layers fine-tuned |
| Pretrained revision | `491b427b45c94c7fb0e78b5474cc919aff584bbf` |
| Input | 224 × 224 RGB, bicubic interpolation, ImageNet normalization |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01 |
| Schedule | One warmup epoch, then cosine decay over at most 20 epochs |
| Batch / seed | 64 / 42 |
| Sampling | Inverse class frequency, with replacement |
| Training augmentation | Random resized crop (scale 0.85–1), both flips, rotation ±10°, brightness/contrast 0.1 |
| Label smoothing | 0.1 during training; none in validation/test |
| Early stopping | Patience 5, minimum validation cross-entropy |
| Hardware | One remote RTX A6000, GPU 1 at launch |
| Precision | bfloat16 autocast for training, float32 validation |
| Loading | Four spawned loader workers, two main-process CPU threads |
| Completed / selected epoch | 14 / 9 |
| Stop reason | `early_stopping` |
| Successful optimizer updates | 1232 |
| Training wall time | 6.28 minutes, including model initialization and validation |

ResNet-18 establishes a small CPU-serving baseline before trying larger backbones.
Weighted sampling gives each class equal expected training representation; it
repeats some images and omits others per epoch. We do not also weight the loss.
Both flips assume casting orientation does not determine the defect label.
Training loss uses augmentation, resampling and label smoothing; it is not directly
comparable to the ordinary validation/test cross-entropy.

### Selected checkpoint results

These reports evaluate the same saved checkpoint on CPU in float32. Training
selection used float32 validation on GPU. The saved checkpoint's validation loss
was independently reproduced on CPU within 0.00001.

| Split | Accuracy | Macro F1 | ROC-AUC | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 99.59% | 0.995862 | 0.999950 | 0.057477 |
| Held-out test | 99.72% | 0.996993 | 0.999890 | 0.056902 |

Test metrics per class:

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| ok | 262 | 99.24% | 100.00% | 0.996198 |
| defective | 453 | 100.00% | 99.56% | 0.997788 |

Test confusion matrix, with true labels on rows and predictions on columns:

| True / predicted | OK | Defective |
| --- | ---: | ---: |
| OK | 262 | 0 |
| Defective | 2 | 451 |

The model misses **2 of 453 defective images** and flags **0 of 262
OK images** as defective. Defective-class recall is **99.56%**,
meeting the plan's 97% target on this split.
ROC-AUC treats `defective` as positive. Softmax confidence is uncalibrated; the
snapshot also records up to 16 most confident test mistakes for later inspection.
The test result did not trigger a training or threshold change.

### ONNX parity and CPU latency

The full export check passed on **32 held-out test images**, each checked at batch
size 1 and again in batches of 8. Maximum absolute logit error was
**2.2649765e-06**, below the required 0.001. The graph is
float32 ONNX opset 17 with dynamic batch size and fixed 224 × 224 input.

| Measurement | Result |
| --- | ---: |
| ONNX Runtime CPU mean | 16.96 ms/image |
| ONNX Runtime CPU p95 | 17.51 ms/image |
| Warmup / measured calls | 10 / 200 |
| Batch / intra-op threads / inter-op threads | 1 / 2 / 1 |

Measured locally on **AMD Ryzen 5 5600H with Radeon Graphics** with ONNX Runtime
1.31.0 and `CPUExecutionProvider`.
Both mean and p95 are within the 50 ms target for this model-only CPU benchmark.
Timing covers `session.run` on an already-preprocessed image; decoding, resizing,
normalization, session startup, HTTP and network overhead are excluded. It is not
a deployed-service latency result, and machine load can affect timings.

### Reproduce

Use the pinned CPU/CUDA environments described in the README. The original run
used the existing cached pretrained revision with `HF_HUB_OFFLINE=1`. On the VM,
select a GPU assigned to your workload and use a new output directory:

```bash
export CUDA_VISIBLE_DEVICES=1  # replace with the GPU selected for your workload
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.train_classifier \
  --manifest /usershome/t25089/visionforge/data/manifests/casting-v2/manifest.json \
  --data-dir /usershome/t25089/visionforge/data \
  --cache-dir /usershome/t25089/visionforge/models \
  --output /usershome/t25089/visionforge/runs/casting-baseline-reproduction \
  --device cuda --epochs 20 --batch-size 64 --learning-rate 0.0003 \
  --weight-decay 0.01 --label-smoothing 0.1 --patience 5 --seed 42 \
  --workers 4 --cpu-threads 2 --horizontal-flip --vertical-flip
```

Copy the completed run's checkpoint, manifest and metadata into local `runs/`.
Then evaluate and export, using a new report/output path when reproducing:

```bash
BASELINE_RUN=runs/casting-baseline-reproduction
BASELINE_PYTHON="$HOME/.local/share/visionforge/venv-cpu/bin/python"
"$BASELINE_PYTHON" -m scripts.evaluate_classifier \
  --checkpoint "$BASELINE_RUN/best.pt" --split val \
  --output "$BASELINE_RUN/validation-report.json"
"$BASELINE_PYTHON" -m scripts.evaluate_classifier \
  --checkpoint "$BASELINE_RUN/best.pt" --output "$BASELINE_RUN/report.json"
"$BASELINE_PYTHON" -m scripts.export_classifier \
  --checkpoint "$BASELINE_RUN/best.pt" --output "$BASELINE_RUN/onnx"
```

Full artifacts for the measured run are retained under
`runs/casting-baseline-20261019/` locally and
`/usershome/t25089/visionforge/runs/casting-baseline-20261019/` on the VM.
The committed JSON records their SHA-256 hashes. Images, checkpoints, and ONNX
binaries are not committed. Exact reproduction assumes the same software,
hardware and worker configuration; CPU and CUDA need not produce identical runs.

## Remaining baseline work

- [x] Full casting training, validation-selected checkpoint, held-out evaluation.
- [x] Casting ONNX parity, local CPU latency, notebook 01 and recorded results.
- [ ] Full NEU-DET training, held-out evaluation, ONNX verification and notebook 02.
- [ ] Complete the baseline milestone before starting the application services.
- [ ] Later: active-learning comparison and notebook 03.
