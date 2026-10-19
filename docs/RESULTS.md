# Baseline results

## Casting classifier — casting-baseline-20261019

This is the first full-data casting baseline, trained on 2026-10-19. It uses
pretrained ResNet-18, the frozen casting-v2 manifest, and the existing training
recipe. Hyperparameters were fixed before test evaluation. The checkpoint was
selected solely by validation loss; classification uses argmax with no test-tuned
threshold. The NEU-DET baseline is recorded below; active learning remains pending.

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

## NEU-DET detector — neu-det-baseline-20261019

The full pretrained YOLOv8s baseline completed on 2026-10-19. The recipe was
fixed before test evaluation; the checkpoint was selected only by validation
mAP@0.5:0.95. Test results did not trigger retraining or threshold tuning.

The [recorded snapshot](results/neu-det-baseline-20261019.json) includes history,
resolved training arguments, CPU evaluation reports, the failed ONNX attempt and
artifact hashes. [Notebook 02](../notebooks/02_neu_detection_baseline.ipynb) reads
that snapshot and shows learning curves and per-class test AP without weights or a GPU.
Training source: [c25334d](https://github.com/sugerdarco/visionForage/commit/c25334daf2dd18d97550943f92bbfe501d61882b).
Evaluation/export source: [97149a6](https://github.com/sugerdarco/visionForage/commit/97149a668210fe0c94483039d5034f8c3d62d081).

### Dataset and training

| Split | Images | Manifest boxes | Effective loader boxes |
| --- | ---: | ---: | ---: |
| Train | 1,259 | 2,909 | 2,908 |
| Validation | 270 | 625 | 625 |
| Test | 271 | 655 | 653 |

All 1,800 image files and XML annotations were checked against the preparation
audit before training. Manifest SHA-256:
`8bf7bfca0af99f0af4decdbdab53a423fcf32ac34822d7cf6d3963c059859615`.
The loader removes one duplicate training label row and two duplicate test rows
(one crazing, one patches), with no other label or image loss. Identical pixels
stay in one split; two test images (`patches_101` and
`patches_105`) have identical pixels but different annotations and are retained.
Difficult/truncated objects remain ordinary targets; the test manifest contains
20 difficult annotation rows. This is a custom split stratified by filename family, not an official
benchmark. No acquisition/parent IDs are available to prove full independence.

| Setting | Value |
| --- | --- |
| Model | COCO-pretrained YOLOv8s, six defect classes |
| Input / batch / seed | 640 × 640 / 16 / 42 |
| Hardware | Remote RTX A6000, GPU 1 at launch |
| Precision / loading | float16 AMP / four workers, two CPU threads |
| Epoch limit / early-stop patience | 100 / 20 |
| Completed / selected epoch | 100 / 96 |
| Selection fitness | 0.43982 validation mAP@0.5:0.95 |
| Resolved optimizer | AdamW, learning rate 0.001, momentum 0.9, weight decay 0.0005 |
| Schedule | Three warmup epochs, linear decay; nominal batch size 64 with accumulation |
| Augmentation | Pinned YOLO defaults for HSV, translation, scale and mosaic; both flips disabled; mosaic closes for final 10 epochs |
| Successful / AMP-skipped updates | 2021 / 10 |
| Training wall time | 15.27 minutes including setup and validation |

YOLOv8s is a compact one-stage baseline that fits the shared manifest-to-ONNX
workflow. The 640-pixel input follows the plan, although source images are
200 × 200: resizing increases compute without adding new source detail. Larger
models or alternative input sizes should be compared on validation before any
new held-out evaluation. Flips remain disabled to preserve surface orientation.

`optimizer=auto` chose AdamW; the saved pre-resolution arguments still contain
library defaults for `lr0` and `momentum`. The snapshot records the actual
optimizer announcement separately. AMP skipped 10 overflowing updates while
adjusting its scale; every epoch made successful updates. Training reached the
100-epoch limit rather than stopping early.

### Selected checkpoint results

These reports evaluate the saved epoch-96 checkpoint on CPU in float32,
with fixed square letterboxing and batch 16. The trainer uses a different
validation path (padding/precision and final model fusion can differ), so its
selection fitness and final validation recheck are retained separately in the
snapshot. The CPU report is not claimed to exactly reproduce those numbers.

AP uses confidence floor 0.001, NMS IoU 0.7, and at most 300 detections/image.
Fixed precision/recall uses confidence 0.25 and matching IoU 0.5, with matches
recomputed after filtering. It is distinct from the trainer's best-F1 threshold.

| Split | mAP@0.5 | mAP@0.5:0.95 | Fixed micro precision | Fixed micro recall |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.7194 | 0.4116 | 60.08% | 74.88% |
| Held-out test | 0.7427 | 0.4105 | 63.91% | 75.65% |

Test per-class object metrics:

| Class | Boxes | AP@0.5 | AP@0.5:0.95 | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| crazing | 105 | 0.3861 | 0.1638 | 38.24% | 49.52% | 84 | 53 |
| inclusion | 157 | 0.7860 | 0.4131 | 72.46% | 77.07% | 46 | 36 |
| patches | 140 | 0.8721 | 0.5463 | 74.85% | 87.14% | 41 | 18 |
| pitted_surface | 63 | 0.9152 | 0.5673 | 76.71% | 88.89% | 17 | 7 |
| rolled-in_scale | 100 | 0.5974 | 0.2600 | 54.78% | 63.00% | 52 | 37 |
| scratches | 88 | 0.8993 | 0.5123 | 67.23% | 90.91% | 39 | 8 |

Test **mAP@0.5 is 0.7427**, meeting the plan's **0.70** target on this
split. At confidence 0.25, there are 494 matched objects,
279 false-positive detections and
159 missed objects. Per-class differences matter;
the overall mean does not guarantee equally reliable detection of every defect.
Crazing is weakest, with AP@0.5 of 0.3861 and recall of 49.52% at confidence 0.25.

### ONNX gate: failed, no bundle published

The strict export attempt requested 32 held-out test images at batches 1 and 8.
It stopped at a failing batch whose maximum absolute error was
**0.001678466796875**, above **0.001**.
This is the first failing batch's error, not the maximum over a completed
32-image check. No ONNX bundle was published, and the CPU latency stage was not
reached. The detector export gate remains open even though test mAP passes.

The candidate graph uses float32, opset 17, input `[batch, 3, 640, 640]` and
output `[batch, 10, 8400]`. Parity covers every raw coordinate in input pixels and
every class probability before confidence filtering/NMS. Relative tolerance is
zero. We did not normalize coordinates, filter predictions, reduce sample count
or change tolerance to obtain a pass.

The current exporter preserves Conv/BatchNorm operations with constant folding
disabled. Its original unfused float32 PyTorch reference uses native CPU kernels
(oneDNN disabled). ONNX Runtime uses its standard graph optimizations. That
configuration passed earlier synthetic checks but is insufficient for this
trained model on real images.

Additional diagnostics used the same checkpoint, first eight seed-42 synthetic
images and first 32 **validation** images; they did not retrain or tune quality on
the test set. All four runtime optimization levels passed the synthetic probe,
but failed on validation. The snapshot records both batch sizes (1 and 8), raw
box/probability errors, and counts outside tolerance. These observations suggest
input-dependent floating-point rounding; they do not establish an acceptable
serving export or justify relaxing the required gate.

The failed attempt, diagnostics, and their hashes are retained for follow-up.
Detector latency is **not measured**. The casting classifier's independent
passing export and CPU latency result remain valid.

### Reproduce

Use the pinned CUDA environment on the VM and a new output directory. Select a
GPU assigned to this workload; the recorded run used GPU 1 when it was idle.

```bash
export CUDA_VISIBLE_DEVICES=1
NEU_RUN=/usershome/t25089/visionforge/runs/neu-det-baseline-reproduction
NEU_PYTHON=/var/tmp/visionforge-t25089/venv-cu126/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$NEU_PYTHON" -m scripts.train_detector \
  --manifest /usershome/t25089/visionforge/data/manifests/neu-det-v1/manifest.json \
  --data-dir /usershome/t25089/visionforge/data \
  --cache-dir /usershome/t25089/visionforge/models/yolo \
  --output "$NEU_RUN" --device cuda --epochs 100 --image-size 640 \
  --batch-size 16 --patience 20 --seed 42 --workers 4 --cpu-threads 2
"$NEU_PYTHON" -m scripts.evaluate_detector \
  --checkpoint "$NEU_RUN/best.pt" --data-dir /usershome/t25089/visionforge/data \
  --split val --output "$NEU_RUN/validation-report.json"
"$NEU_PYTHON" -m scripts.evaluate_detector \
  --checkpoint "$NEU_RUN/best.pt" --data-dir /usershome/t25089/visionforge/data
"$NEU_PYTHON" -m scripts.export_detector \
  --checkpoint "$NEU_RUN/best.pt" --data-dir /usershome/t25089/visionforge/data
```

Training/evaluation artifacts and export diagnostic logs are retained in
`runs/neu-det-baseline-20261019/` locally and
`/usershome/t25089/visionforge/runs/neu-det-baseline-20261019/` on the VM.
The VM also retains the prepared training files and last-epoch weights.
The committed snapshot hashes the retained core artifacts; binary weights and
dataset images stay outside Git. This run has no published ONNX bundle. Exact
reproduction depends on the pinned software, hardware, worker count and source
revisions.

## Baseline progress

- [x] Full casting training, validation-selected checkpoint, held-out evaluation.
- [x] Casting ONNX parity, local CPU latency, notebook 01 and recorded results.
- [x] Full NEU-DET training, held-out evaluation and notebook 02 with recorded export failure.
- [x] Both held-out model quality targets passed on the frozen splits.
- [ ] Next: resolve detector raw-output ONNX parity and measure CPU latency.
- [ ] Finish the baseline export gate before local application services.
- [ ] Later: active-learning comparison and notebook 03.
