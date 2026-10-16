# VisionForge

A visual inspection platform for industrial defects. The planned workflow is to
upload images, label defects, train a model, and deploy it for inference.

The first milestone is reproducible classification and detection baselines.
The application and deployment services will follow once the ML pipeline works.
No model results have been measured yet.

## Development setup

Lightweight development and CPU checks run locally. Training runs on a remote
Linux GPU host through SSH, using an isolated Python 3.11 environment without
sudo. Docker Compose is planned for the local application services; remote
training will not require Docker access.

Check either machine before installing dependencies:

```bash
bash scripts/check_environment.sh
ssh user@gpu-host 'bash -s' < scripts/check_environment.sh
```

To inspect an existing PyTorch installation as well:

```bash
bash scripts/check_environment.sh --torch
ssh user@gpu-host 'bash -s -- --torch' < scripts/check_environment.sh
```

The script reports missing tools without installing anything. It is an inventory,
not a pass/fail deployment check. The optional PyTorch probe checks driver access
but does not run training or prove that CUDA kernels work.

## Training environment

From a checkout on either Linux x86_64 machine, create a dedicated environment:

```bash
# Local CPU development
bash scripts/setup_training_env.sh "$HOME/.local/share/visionforge" cpu
"$HOME/.local/share/visionforge/venv-cpu/bin/python" scripts/check_torch.py

# Remote GPU host
bash scripts/setup_training_env.sh "/var/tmp/visionforge-$USER" cu126
# Set this to a GPU allocated to you; availability alone is not a reservation.
CUDA_VISIBLE_DEVICES=1 "/var/tmp/visionforge-$USER/venv-cu126/bin/python" \
  scripts/check_torch.py --device cuda
```

The setup requires Bash, curl, tar, coreutils, and network access. It verifies the
uv archive's SHA-256 checksum, then installs uv 0.12.19, Python 3.11.16, and a
virtual environment inside the supplied directory.
It does not modify shell startup files or install packages into system Python.
Rerunning it synchronizes the pinned requirements and installs the shared `vf_ml`
package from the checkout in editable mode. Other installed packages are removed;
use this environment only for VisionForge.

The GPU example uses scratch storage. Its contents can be recreated, but the
directory may be cleaned by the host; keep datasets and experiment results in
persistent storage. Choose another absolute path if scratch storage is unsuitable.

Why these choices:

- **uv + Python 3.11:** uv downloads a prebuilt interpreter without sudo. This
  keeps local and remote code on the same Python version despite different system
  installations. [Python installation documentation](https://docs.astral.sh/uv/guides/install-python/).
- **PyTorch 2.13.0 + torchvision 0.28.0:** an officially paired release for tensor
  training and image operations. CPU wheels keep local dependencies smaller;
  CUDA 12.6 wheels supply the remote GPU runtime. The NVIDIA driver must already
  be installed; this setup does not install it or a system CUDA toolkit.
  [Official version pairs](https://pytorch.org/get-started/previous-versions/).
- **Separate environments:** Docker is unavailable to the remote account, while
  a user-owned Python environment works without daemon access. It isolates Python
  dependencies but does not provide container-level operating-system isolation.

The smoke test checks CPU/GPU numerical agreement, convolution, mixed precision,
backpropagation, an optimizer step, and torchvision NMS. CUDA mode requires an
explicit single GPU selection and fails rather than silently falling back to CPU.
It verifies runtime compatibility, not model accuracy or training throughput.

Run the offline tests with `make test PYTHON="$HOME/.local/share/visionforge/venv-cpu/bin/python"`;
it includes the local package on the import path and uses the pinned image decoder.
The dependency lists are generated from `requirements/torch.in`, including
transitive dependencies, separately for each backend:

```bash
uv pip compile requirements/torch.in --python python3.11 \
  --index https://download.pytorch.org/whl/cu126 --default-index https://pypi.org/simple \
  --no-annotate --no-header -o requirements/torch-cu126.txt
# Repeat with cpu in place of cu126 to update the CPU list.
```

The PyTorch index has priority for its CPU/CUDA wheels; PyPI supplies `timm` and
packages absent from that index. All resolved versions are pinned. When adding a
dependency, use the existing resolved list as `--constraint` and write to a temporary
output before replacing it, keeping unrelated packages unchanged.

## Casting dataset

```bash
python3 scripts/download_datasets.py casting
# An existing copy of the same archive also works, without network access:
python3 scripts/download_datasets.py casting --archive /path/to/casting-v2.zip
```

Use `--data-dir /persistent/path/to/data` on the remote host. Archives go in
`data/downloads/` and extracted images in `data/raw/casting-v2/`. Neither is
committed. The downloader uses Python 3.11+ standard-library modules, so it needs
no GPU, PyTorch, or Kaggle client installation. The public download currently
works without credentials; `--archive` supports downloading through Kaggle's
website if its access requirements change.

The source is Ravirajsinh Dabhi's
[casting product image dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product),
version 2, listed under **CC BY-NC-ND 4.0**. See
[datasets/casting.json](datasets/casting.json) for attribution, source, sizes,
folder counts, and the SHA-256 measured from our download. This is a project
checksum, not an independently published checksum from the dataset author.

The archive contains 7,348 images under `casting_data/casting_data/` (6,633 train,
715 test), plus 1,300 originals under `casting_512x512/casting_512x512/`. The source
says the 300×300 images already include augmentation. These collections stay
separate: the downloader preserves the supplied folders and does not create new
splits. The preparation command below checks exact duplicates; related augmented
images can still overstate generalization. Inspection of all
8,648 image headers found RGB JPEGs, despite their grayscale appearance.

Downloads are verified before extraction. Incomplete downloads and failed
extractions are discarded; existing archives and extracted files are checked on
reruns. Changed files cause an error instead of being silently overwritten. The
tradeoff is a small amount of disk I/O on each rerun for reproducible inputs.

Run `make test` for the offline tests and `make lint` for Python lint and shell
syntax checks. Lint uses Ruff 0.14.1 through uv, pinned to keep checks consistent
without adding a development tool to the training environment.

### Prepare casting splits

From the repository root, after setting up the training environment:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.prepare_casting
# On the VM, from its source checkout:
/var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.prepare_casting \
  --data-dir /usershome/t25089/visionforge/data
```

The command reuses the downloader to verify the pinned archive and extracted files,
then decodes the 300×300 images with Pillow. It needs only CPU and user-writable
storage, so the same preparation works locally and on the VM without sudo. Pillow
is already pinned in both training environments; no extra image-processing stack
is needed. The separate 512×512 originals are excluded.

Our version 2 audit found **64 exact duplicate pairs crossing the supplied train
and test folders**, all in `ok_front`, with no conflicting labels. We preserve all
715 supplied test images and exclude their 64 training copies. Remaining training
duplicates, if present, are reduced to one image; conflicting labels stop preparation.
Both file SHA-256 and decoded RGB pixel SHA-256 are recorded, so changing JPEG
metadata cannot hide an exact pixel duplicate.

With seed `42`, 15% of each remaining training class becomes validation, rounded
to the nearest image. Selection ranks a SHA-256 of the seed and pixel digest,
making it independent of directory ordering. Each class retains at least one
training and validation image. Preserving the supplied test set gives continuity
with the source split; the tradeoff is that the final ratios differ from 70/15/15.

| Split | ok (0) | defective (1) | Total |
| --- | ---: | ---: | ---: |
| Train | 2,389 | 3,194 | 5,583 |
| Validation | 422 | 564 | 986 |
| Test | 262 | 453 | 715 |

Outputs go to `DATA_DIR/manifests/casting-v2/` and stay out of Git:

- `manifest.json` uses the shared `vf_ml.manifest` contract. Image keys are relative
  to `DATA_DIR`, allowing the same manifest to resolve images on either machine
  and later become object-storage keys for training workers.
- `report.json` records source metadata/checksum, seed, split method, decoder
  version, per-image checksums and assignments, duplicate exclusions, counts, and
  the manifest checksum. It supports auditing without changing the training contract.

Rerunning verifies identical output; changed settings or files cannot silently
replace the snapshot. Use a new `--output` directory when changing `--seed` or
`--val-fraction`. Source images are never moved or deleted.

**Evaluation limitation:** these images were already augmented upstream, and this
snapshot has no parent-image mapping. Exact duplicate checks do not detect rotated,
cropped, or otherwise altered relatives. Related-image leakage remains unresolved;
the split supports an initial baseline, not a claim of independent production
generalization. Investigate parent groups before making that claim.

## NEU-DET dataset

```bash
python3 scripts/download_datasets.py neu-det
# The same --data-dir and --archive options work for this dataset.
```

This uses the NEU-DET ZIP linked by Kechen Song and Yunhui Yan on the
[authors' dataset page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/).
We use the original archive because it keeps image/annotation pairs together
without introducing a third-party train/validation split. The authors' page
requests citation but does not state an explicit dataset license.

Files are preserved under `data/raw/neu-det-v1/NEU-DET/IMAGES/` and
`ANNOTATIONS/`. Here `v1` means our first checksum-pinned snapshot; the upstream
download has no published version number. Source details and measured counts
are recorded in [datasets/neu-det.json](datasets/neu-det.json).

The snapshot contains 1,800 RGB JPEGs at 200×200 pixels, 1,800 Pascal VOC XML
annotations, and 4,189 boxes across six classes. A full audit decoded every
image, matched image/XML file stems, checked dimensions, and found all boxes
within the image bounds. Two details matter when importing these annotations:

- 174 XML `filename` fields omit the image extension. Pair image and annotation
  files by their shared stem instead of treating this field as a full path.
- 123 images contain multiple object classes. Read each object's `name` from
  the XML; the image filename's class prefix is not the label for every box.

XML is allowed only when declared in a dataset's metadata. Existing checksum,
safe extraction, and rerun verification checks apply to annotations as well as
images. This step does not convert labels to YOLO format or create new splits;
those belong in the manifest and training adapters.

## Dataset manifest

`vf_ml.manifest` is the shared contract between dataset preparation and training.
It loads JSON into immutable records using only the Python standard library.
Workers will consume these snapshots instead of querying changing database rows.
JSON keeps the inputs inspectable and portable; strict validation catches mistakes
before a GPU job starts. Unknown fields and unsupported schema versions fail
explicitly so that typos or future format changes cannot be silently ignored.

A minimal classification manifest looks like this (the key is illustrative):

```json
{
  "schema_version": 1,
  "task_type": "classification",
  "class_map": {"ok": 0, "defective": 1},
  "samples": [
    {
      "image_key": "project/image-001.jpg",
      "split": "train",
      "labels": [{"class_index": 1, "bbox": null}]
    }
  ]
}
```

- `class_map` assigns unique, contiguous indices starting at zero. JSON field
  order does not determine model output indices.
- `image_key` is a normalized relative key within a configured image bucket or
  local data root. Absolute paths, URLs, traversal, and repeated keys are rejected.
- `split` is `train`, `val`, or `test`. A manifest must contain images, but need
  not contain every split; the training entry point will require its needed splits.
- Classification has exactly one label and a null box. Detection uses the same
  structure with `task_type: "detection"` and boxes such as
  `{"class_index": 1, "bbox": [0.1, 0.2, 0.3, 0.4]}`. Coordinates are normalized
  **top-left x, top-left y, width, height**, not YOLO center coordinates. Boxes
  need positive area and must fit inside the image (edge tolerance: 1e-9).
  An empty detection label list means an annotated image with no objects.

After running the environment setup from this checkout, inspect a manifest with:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m vf_ml.manifest /path/to/manifest.json
```

Or load it in Python:

```python
from vf_ml.manifest import load_manifest

manifest = load_manifest("manifest.json")
training_samples = manifest.samples_for("train")
```

The manifest loader reads only JSON. The casting preparation command handles
checksums, duplicate auditing, and split creation; the dataset below decodes images.
Immutability protects the loaded records; it does not freeze the files referenced
by their keys or prevent changes to the JSON on disk.

## Classification images and transforms

`ClassificationDataset` connects the frozen manifest to PyTorch's `DataLoader`.
It reads only the selected split, preserves the manifest's class indices, and
opens images lazily on CPU. No folder scan, database, or GPU is needed. Missing
or corrupt images fail with their image key; symbolic links outside the configured
data root are rejected. Source integrity is checked during dataset preparation,
not by hashing every image again in every training epoch.

Run this with the prepared Python environment from the repository root:

```python
import torch
from torch.utils.data import DataLoader
from vf_ml.classification import ClassificationDataset
from vf_ml.manifest import load_manifest
from vf_ml.transforms import ClassificationTransforms

torch.manual_seed(42)
manifest = load_manifest("data/manifests/casting-v2/manifest.json")
config = ClassificationTransforms(horizontal_flip=True, vertical_flip=True)
dataset = ClassificationDataset(manifest, "data", "train", config)
loader = DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(42),
)
batch = next(iter(loader))
print(batch["image"].shape)  # [16, 3, 224, 224], float32 on CPU
print(batch["target"])      # int64 class indices: ok=0, defective=1
print(batch["image_key"])   # keys for tracing predictions and mistakes
```

On the VM, replace `data` with `/usershome/t25089/visionforge/data` in both paths.
Start with `num_workers=0` for straightforward debugging and low memory use locally.
Multiple workers can overlap decoding with training; they also consume more RAM.
Spawned workers are tested, but scripts using them need an `if __name__ == "__main__":`
guard. Seed the loader generator for worker randomness and shuffling; with zero
workers, augmentation uses the main process's PyTorch RNG. Reproducibility assumes
the same pinned software, seed, and worker configuration.

We use the already-installed [torchvision v2 transforms](https://docs.pytorch.org/vision/stable/transforms.html)
to avoid an additional augmentation dependency. The shared pipeline converts PIL
images (including grayscale) to three RGB channels, then scales pixels to `[0, 1]`
and applies configurable ImageNet mean/std normalization. Defaults are 224 pixels,
mean `(0.485, 0.456, 0.406)`, and std `(0.229, 0.224, 0.225)`.

- **Training:** random resized crop (85–100% area, aspect ratio 0.95–1.05),
  rotation up to 10 degrees, and brightness/contrast jitter of 0.1. Horizontal and
  vertical flips are disabled by default; the casting example enables both.
  Orientation-sensitive projects should keep them disabled. Crops can remove
  small defects and rotations introduce border pixels, so these settings need
  validation when we train the baseline.
- **Validation/test:** fixed resize of the full image, with antialiasing
  and no random augmentation. This preserves edge content that center cropping
  could remove. It preserves the shape of square casting images but stretches
  non-square inputs; this is an explicit task-specific choice.

`classification_transform(config)` exposes the same deterministic path for future
inference. Store `dataclasses.asdict(config)` with model artifacts and reconstruct
it with `ClassificationTransforms(**saved_config)` to reuse preprocessing.
Normalization and interpolation must match the selected pretrained model; the
model factory below checks these settings. Standalone transforms default to
bilinear, while this ResNet-18 checkpoint requires bicubic resizing.
The lightweight `vf_ml.manifest` module remains independent of PyTorch; API code
should import that module, not the classification modules.

## ResNet-18 classification baseline

`vf_ml.models.create_classifier` connects the manifest and shared transforms to
[`timm`'s pretrained ResNet-18](https://huggingface.co/timm/resnet18.a1_in1k).
We pin `timm==1.0.30` and checkpoint revision
`491b427b45c94c7fb0e78b5474cc919aff584bbf`, using `model.safetensors`.
The backbone starts with ImageNet weights; its classifier is replaced with a new,
untrained head whose outputs follow `manifest.class_names`. Casting therefore
has 11,177,538 trainable parameters and two raw logits per image, ordered `ok`,
`defective`. These are not calibrated probabilities or trained defect predictions.

ResNet-18 is the plan's small first baseline: practical CPU checks locally and
straightforward fine-tuning on one remote GPU. `timm` also exposes alternatives
such as EfficientNet and ConvNeXt through a common interface, but this factory
deliberately supports just the selected baseline until we have measured results.
Using torchvision's model directly would avoid `timm`'s extra dependencies;
following the plan's model library makes later comparisons easier.

```python
from vf_ml.models import create_classifier

classifier = create_classifier(manifest, cache_dir="data/model-cache")
dataset = ClassificationDataset(manifest, "data", "train", classifier.transforms)
model = classifier.model  # CPU, training mode; move to the selected device in training
# Save classifier.metadata with future checkpoints: weights, classes, transforms.
```

The factory derives 224×224 input size, ImageNet normalization, and bicubic
interpolation from the model configuration. Passing an incompatible transform
configuration raises an error before fetching weights. Custom augmentation is
supported by passing matching settings, for example
`ClassificationTransforms(interpolation="bicubic", horizontal_flip=True)`.
We retain full-image evaluation resizing for edge defects instead of the source
checkpoint's center crop (`crop_pct=0.95`); metadata records both configurations.
This is a fine-tuning choice, not a reproduction of its ImageNet benchmark.

Rerun the environment setup to install the updated pinned dependencies, then:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.check_classifier \
  --cache-dir data/model-cache
# On the VM, from the source directory; this also defaults to CPU:
/var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.check_classifier \
  --data-dir /usershome/t25089/visionforge/data \
  --cache-dir /usershome/t25089/visionforge/models
```

The check uses four training images for a forward pass, finite loss and gradient
checks, and one optimizer update. It does not save a trained checkpoint or report
accuracy. Add `--device cuda` only after selecting one available GPU with
`CUDA_VISIBLE_DEVICES`. This step was verified on CPU locally and inside the VM's
CUDA environment; the shared VM GPUs were busy, so GPU model execution is pending.

The first pretrained run downloads about 45 MB into the chosen user-writable
cache. Reuse that cache with `HF_HUB_OFFLINE=1` for offline runs. A failed download
stops construction; it never silently falls back to random weights. Explicit
`--random-init` (or `pretrained=False` in Python) supports offline wiring tests and
records that no pretrained weights were loaded. Model weights stay out of Git.
The next step is the training loop; no casting accuracy has been measured yet.

## Planned service connections

The web app calls the FastAPI API. PostgreSQL stores metadata, and S3-compatible
storage holds images, frozen dataset manifests, and model artifacts. The API
queues training jobs through Redis and Celery. Remote workers read a manifest,
train a model, and return metrics and artifacts. CPU inference containers serve
exported ONNX models behind a reverse proxy.

Private service connections between the machines will use SSH forwarding. This
avoids exposing database and queue ports, but requires the local machine and
tunnel to remain available during integrated training. The initial ML baseline
will run directly on the remote host without these application services.
