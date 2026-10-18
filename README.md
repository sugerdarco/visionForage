# VisionForge

A visual inspection platform for industrial defects. The planned workflow is to
upload images, label defects, train a model, and deploy it for inference.

The first milestone is reproducible classification and detection baselines.
The application and deployment services will follow once the ML pipeline works.
Full-dataset baseline results are still pending; small execution checks are documented below.

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
images. The preparation step below creates our frozen splits; conversion to YOLO
training files belongs in the next adapter.

## Prepare NEU-DET detection splits

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.prepare_neu
```

On the VM, run from `/var/tmp/visionforge-t25089/source`:

```bash
/var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.prepare_neu \
  --data-dir /usershome/t25089/visionforge/data
```

This verifies the pinned archive and extracted files, decodes every image, and
writes `DATA_DIR/manifests/neu-det-v1/{manifest,report}.json`. It needs only the
existing Python/Pillow environment, no GPU or sudo. Identical reruns are accepted;
changed settings or contents require a new `--output` directory. Source images,
XML files, and generated snapshots stay outside Git.

The connection is **source JPEG + XML → shared detection manifest → future YOLO
adapter**. Each manifest row contains a relative image key, fixed split, and all
of that image's object labels. Class indices are stable: `crazing=0`, `inclusion=1`,
`patches=2`, `pitted_surface=3`, `rolled-in_scale=4`, `scratches=5`. XML object names
supply the labels; filename prefixes only help balance the split.

With seed 42 and 15% validation / 15% test pixel groups, the verified result is:

| Split | Images | Unique pixel groups | Boxes | Difficult boxes included |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,259 | 1,259 | 2,909 | 51 |
| Validation | 270 | 270 | 625 | 10 |
| Test | 271 | 270 | 655 | 20 |

All 1,800 images and 4,189 boxes are preserved, and each split contains all six
object classes. Within each filename family, groups of identical RGB pixels are
ranked by SHA-256 of seed and pixel hash. Rounded test and validation group counts
are selected first; the rest train. This avoids dependence on filesystem ordering
and requires no additional splitting library. It balances source families rather
than every combination of object labels, so the report includes per-class image
and box counts. `--seed`, `--val-fraction`, and `--test-fraction` are configurable;
freeze them before training and never adjust them in response to test scores.

Two audited details affect interpretation:

- `patches_101.jpg` and `patches_105.jpg` have identical pixels but slightly different
  boxes. Both are retained in the **same test split**. We preserve the original
  annotations rather than arbitrarily choose one; they count as two evaluation
  images, not two independent scenes. No exact pixel group crosses a split.
- All 81 `difficult=1` objects and all truncated objects remain ordinary targets.
  Removing their boxes would leave visible defects unlabelled. The report retains
  the flags, but manifest schema 1 has no ignore flag. Consequently future metrics
  using this snapshot will include difficult objects, unlike VOC evaluation that
  ignores them; they must be described as results on our custom split.

Coordinates are interpreted as one-based inclusive VOC pixels, consistent with
[Detectron2's VOC loader](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/pascal_voc.py).
For width `W` and height `H`, conversion is
`[(xmin-1)/W, (ymin-1)/H, (xmax-xmin+1)/W, (ymax-ymin+1)/H]`.
The manifest stores normalized **top-left x/y plus width/height**, not YOLO center
coordinates. A full-image box becomes `[0, 0, 1, 1]`, and a one-pixel box remains
positive-area. This convention is explicit because XML-to-YOLO converters differ
on endpoint handling; the later adapter must not apply another pixel offset.

The XML files report depth 1 although the decoded JPEGs are RGB; both observations
are preserved without rejecting valid image/annotation pairs. Missing pairs,
unknown classes, malformed boxes, dimension mismatches, and ambiguous required XML
fields fail preparation. The report records source provenance, image/XML/pixel
hashes, original coordinates and flags, duplicate groups, and the manifest hash.
Local and VM preparation produced byte-identical manifests and reports.

This is a reproducible project split, not an official benchmark partition. Exact
pixel grouping cannot rule out related crops or acquisition sequences because
parent/acquisition IDs are unavailable. Detection training is available below;
held-out detection evaluation remains to be implemented.

## Prepare YOLO training files

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.prepare_yolo \
  --manifest data/manifests/neu-det-v1/manifest.json \
  --data-dir data --output data/yolo/neu-det-v1-train
```

On the VM, run from `/var/tmp/visionforge-t25089/source` with its Python environment:

```bash
/var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.prepare_yolo \
  --manifest /usershome/t25089/visionforge/data/manifests/neu-det-v1/manifest.json \
  --data-dir /usershome/t25089/visionforge/data \
  --output /usershome/t25089/visionforge/data/yolo/neu-det-v1-train
```

`vf_ml.yolo.prepare_yolo_dataset` connects our shared manifest to the
[Ultralytics YOLO dataset format](https://docs.ultralytics.com/datasets/detect/).
It writes the following layout using the existing Python/Pillow environment,
without importing PyTorch or Ultralytics or needing GPU/sudo access:

```text
neu-det-v1-train/
  dataset.yaml
  manifest.json
  report.json
  images/train/…jpg
  images/val/…jpg
  labels/train/…txt
  labels/val/…txt
```

Only train and validation images are opened by default. Our NEU-DET snapshot
produces **1,259 train images / 2,909 boxes** and **270 validation images / 625 boxes**.
The 271 test images are omitted from the layout and YAML. For later held-out
evaluation, prepare a separate output with `--include-test`; this adds the test
split without changing its membership. Training and validation must be nonempty,
and every declared class must have a training box.

Each label line contains `class_index x_center y_center width height`. Conversion
from our normalized top-left boxes is only `[x+w/2, y+h/2, w, h]`; no further VOC
pixel offset is applied. Class order, all manifest boxes, and split membership are
preserved. Background images receive explicit empty label files. Label values use
17 significant digits to avoid unnecessary rounding of small boxes.

Images are copied byte-for-byte rather than linked. This uses extra disk space,
but later source changes cannot alter the prepared copies, and training caches or
image repairs cannot change the original dataset. Filenames use the SHA-256 of the
full image key, so equal basenames from different folders remain separate. The
report maps each generated image/label file back to its original key and records
both file hashes, dimensions, counts, and the exact manifest hash. The original
manifest bytes are also included for provenance, including rows for omitted splits.

Preparation validates JPEG/PNG decoding, RGB or grayscale mode, dimensions of at
least 10×10, and that resolved source files stay inside the data root. Images with
non-default EXIF orientation are rejected because rotating pixels alone could
misalign their boxes. No resizing or augmentation happens here; those belong to
the training loader.

The command reserves a new output directory and stages the dataset before
publishing the completed tree. Existing directories are refused, including empty
ones. Handled failures remove the reservation and staging files; a killed process
may leave an incomplete reservation, which must not be used. `dataset.yaml` has an
absolute root for predictable resolution by Ultralytics. Rebuild the layout at its
new destination when moving between machines; the report and generated labels
are independent of that root path.

The real layouts were generated and audited locally and on the VM. Their manifests,
reports, images, and labels match; only the YAML root path differs. The pinned Ultralytics loader is now checked by the training wrapper below,
including its removal of repeated label rows.

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
The training loop below uses this same model factory. A full casting baseline has
not been measured yet.

## Train the classifier

`vf_ml.training.train_classifier` connects the frozen manifest, image loaders,
shared transforms, and ResNet-18 to an optimizer. Training uses only `train` and
`val`; it never opens test images. The initial defaults follow the build plan:
20 epochs, batch size 64, AdamW at `3e-4`, one epoch of linear warmup followed by
cosine decay per batch, label smoothing 0.1, and early stopping after five epochs
without a lower validation loss. Weight decay is 0.01. All layers are fine-tuned.

From the repository root on the VM, after selecting an available allocated GPU
with `CUDA_VISIBLE_DEVICES`:

```bash
/var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.train_classifier \
  --data-dir /usershome/t25089/visionforge/data \
  --manifest /usershome/t25089/visionforge/data/manifests/casting-v2/manifest.json \
  --cache-dir /usershome/t25089/visionforge/models \
  --output /usershome/t25089/visionforge/runs/casting-baseline-1 \
  --device cuda --workers 2 --horizontal-flip --vertical-flip
```

Use `--device cpu` for local checks with the local Python environment and data
paths. CPU execution defaults to two threads and no loader subprocesses. A full
CPU run is supported but heavier than the small checks we run locally. No Docker
or sudo is required on the VM. Use a new output directory for every run; existing
directories are refused, including failed or interrupted runs. `--help` lists
seed, learning-rate, batch-size, epoch, worker, and early-stopping options.

The important choices are:

- **Class imbalance:** sample with replacement using inverse training-class
  frequencies. Each class has equal expected sampling probability, without also
  weighting the loss. This repeats some images and omits others within an epoch;
  class-weighted loss would instead preserve one visit per image. Validation
  retains its original distribution and uses ordinary, unsmoothed cross-entropy.
- **Precision:** CPU uses float32. CUDA training uses autocast with bfloat16 when
  supported, otherwise float16 with gradient scaling. Validation stays float32
  for checkpoint selection. Overflowed float16 updates are skipped and successful
  updates are counted. An epoch with no successful updates fails. Non-finite loss
  also stops the run.
- **Reproducibility:** Python, NumPy, PyTorch, sampling, and data workers are seeded.
  Deterministic PyTorch algorithms are enabled; unsupported deterministic
  operations fail rather than silently changing behavior. Reproducibility assumes
  the same software, hardware, and worker configuration, not identical results
  across CPU/CUDA or different worker counts.
- **Selection:** save a checkpoint only when validation loss improves. Training
  metrics use augmented, resampled data and a smoothed loss, so their loss values
  are not directly comparable with validation loss. Every epoch records accuracy
  and recall for each class; recall is null if a split has no examples of that class.

Each run produces these local artifacts, all outside Git:

| File | Purpose |
| --- | --- |
| `manifest.json` | Exact snapshot used by this run |
| `run.json` | Settings, seed, manifest SHA-256, model/transform metadata, precision, device, status, best epoch, stop reason |
| `history.json` | Epoch train/validation loss, accuracy, per-class recall, learning rate, successful optimizer steps |
| `best.pt` | CPU model state, class/transform metadata, training settings, manifest hash, selected epoch and validation loss |

Metrics and best weights are published with temporary files and renames. A caught
error marks the run failed; a forcibly killed process may leave status `running`.
The checkpoint can be loaded with `torch.load(..., weights_only=True)` and feeds
the evaluator and ONNX exporter below. It is an evaluation checkpoint, not a resumable
training snapshot: optimizer and RNG states are not saved yet.

Verification used a separate manifest of **8 training and 4 validation casting
images**, two epochs, and the actual pretrained model on CPU locally and on the
VM (two spawned workers remotely). This checks execution and artifact creation,
not baseline quality. GPU mixed precision and a full-data baseline remain pending
because the shared GPUs were occupied. The known augmented-relative leakage
limitation still applies to the casting split.

## Evaluate a saved checkpoint

`vf_ml.evaluation.evaluate_checkpoint` loads the saved weights without downloading
a pretrained model or running an optimizer. It checks the manifest's exact SHA-256,
class order, model name, `timm` version, and complete transform settings before
scoring images. Strict state-dictionary loading catches incompatible weights.
Only the chosen `val` or `test` split is opened, using deterministic preprocessing
and model evaluation mode. The original training snapshot must include that split.

For a completed full baseline, the default is the held-out **test** split:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.evaluate_classifier \
  --checkpoint runs/casting-baseline-1/best.pt --data-dir data
```

The command reads `manifest.json` beside the checkpoint and writes `report.json`
there. Override paths with `--manifest` or `--output`. On the VM use its Python
environment and persistent data/run paths. CPU evaluation requires no sudo or
GPU; `--device cuda` requires one explicitly selected, allocated GPU. Existing
reports are never overwritten, and failed evaluations publish no partial report.

For our small execution-check checkpoint, use its **validation** split explicitly:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.evaluate_classifier \
  --checkpoint runs/casting-loop-smoke-20261017/best.pt \
  --split val --batch-size 4 \
  --output runs/casting-loop-smoke-20261017/validation-report.json
```

That smoke run and subset manifest were generated locally and on the VM, not
committed. Its report is labelled `val`; its four-image results are execution
checks, not a full test benchmark. This step leaves the full casting test set
untouched. Test metrics should be computed after model selection, not used to
tune training settings.

The report includes checkpoint/manifest hashes, selected epoch, original model
metadata, runtime settings, and these metrics:

- Accuracy, unsmoothed cross-entropy, per-class precision/recall/F1 and support,
  and macro F1 across every manifest class. Zero-denominator precision/recall/F1
  are reported as zero, with support counts for interpretation. Training history
  uses null for absent-class recall; evaluation uses the documented zero convention.
- A confusion matrix with true classes as rows and predictions as columns,
  labelled in manifest class order.
- ROC-AUC from probabilities. Binary AUC uses class index 1 (`defective` for
  casting); multiclass AUC is the macro one-vs-rest average. A class without both
  positives and negatives has null AUC, and a multiclass macro AUC is null if any
  constituent class is undefined. Tied scores receive half credit.
- Up to 16 mistakes ranked by predicted-class probability, then image key for
  ties. Each includes its image key, true label, predicted label, and confidence,
  ready for the later review UI. Confidence is an uncalibrated softmax score.

Metrics use the already-pinned NumPy dependency. The formulas and averaging
conventions follow the standard definitions documented for
[precision/recall/F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
and [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
Avoiding an additional metrics package keeps this environment smaller; the
tradeoff is maintaining our own focused tests. AUC is checked against independent
positive/negative pair comparisons, including ties and missing classes.
Evaluation keeps class probabilities in memory for exact AUC; this is modest for
the current datasets but would need revisiting for very large evaluations.
The model runs in float32; loss calculation uses float64 to reduce aggregation
rounding. ONNX parity and CPU latency are recorded separately by the exporter below.

## Export a classifier to ONNX

The next connection is `best.pt` + frozen manifest → verified `model.onnx` bundle
→ the later CPU inference service. Export and evaluation share the same checkpoint
loader, so both enforce the manifest hash, class order, model version, transforms,
and strict weight loading. Export runs on CPU locally or on the VM; it needs
neither sudo nor an allocated GPU. Rerun `setup_training_env.sh` to install the
pinned ONNX and ONNX Runtime packages.

For a completed full baseline:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.export_classifier \
  --checkpoint runs/casting-baseline-1/best.pt --data-dir data
```

By default, this reads `manifest.json` beside the checkpoint and creates a new
`onnx/` directory there. `--manifest` and `--output` override these paths. Existing
outputs are refused. Files are staged until all checks pass; `verification.json`
is published last and marks a complete bundle. A process killed during publication
can leave an incomplete directory, which must not be served.

| Artifact | Purpose |
| --- | --- |
| `model.onnx` | Float32 classifier, opset 17, dynamic batch dimension |
| `metadata.json` | Model/checkpoint/manifest hashes, class map, preprocessing and input/output contract |
| `verification.json` | Parity results, selected image keys, CPU latency and runtime details |

The input is **already preprocessed** float32 RGB data with shape `[N, 3, 224, 224]`.
Deterministic full-image resize, bicubic interpolation, and ImageNet normalization
come from `vf_ml.transforms.classification_transform(training=False)` using the
checkpoint settings. Preprocessing is **outside the graph**. Outputs are raw logits
in manifest class order; apply softmax over classes for probabilities. Keeping the
contract beside the graph prevents the inference service from guessing class order
or normalization. A future PyTorch-free serving environment still needs an equivalent
preprocessing implementation and its own parity check.

The export gate checks the ONNX graph and exact opset, then compares raw logits
against CPU PyTorch on the first **32 distinct test images in manifest order**.
Every selected image is checked at batch size 1 and again in batches of up to 8.
The maximum absolute difference must be at most `1e-3`, with no relative tolerance;
invalid shapes, non-finite outputs, insufficient images, or disagreement fail export.
This checks numerical conversion, not predictive accuracy or model quality.

For the existing small execution-check checkpoint, explicitly use validation:

```bash
"$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.export_classifier \
  --checkpoint runs/casting-loop-smoke-20261017/best.pt \
  --split val --parity-samples 4
```

Smaller checks require at least two images to exercise dynamic batching. The report
sets `meets_32_test_image_check` to false for this smoke run; it does not silently
claim the full gate passed. No training or test images are opened in this example.
Our full casting test set remains untouched.

Latency uses `CPUExecutionProvider`, two intra-op threads by default, one inter-op
thread, 10 warmups, then 200 timed single-image `session.run` calls. It repeats the
first selected preprocessed image. Mean and p95 exclude decoding, resizing,
normalization, session initialization, and HTTP overhead. Thread count, CPU model,
platform, and library versions are saved to make the measurement interpretable.
These are single-process measurements on shared machines, not service benchmarks.

Execution checks on 2026-10-17 used each machine's own two-epoch smoke checkpoint:

| Machine | Max absolute logit error | Mean / p95 CPU time |
| --- | --- | --- |
| Local Ryzen 5 5600H | `3.54e-7` | 16.64 / 17.01 ms |
| VM EPYC 9124 | `4.17e-7` | 17.65 / 20.83 ms |

Both used four validation images and batch sizes 1 and 4. Full-baseline accuracy
and the real 32-test-image export gate remain pending. Synthetic tests separately
exercise the 32-image gate, dynamic batching, invalid outputs, and cleanup after
parity failure. Model binaries and generated reports remain outside Git.

Why ONNX Runtime: the planned service can execute the portable graph on CPU without
loading the training model factory or downloading pretrained weights. Its explicit
[CPU execution provider and session API](https://onnxruntime.ai/docs/api/python/api_summary.html)
fit our local Docker deployment and unprivileged VM. We currently select PyTorch's
TorchScript exporter explicitly (`dynamo=False`) for direct opset-17 export and
`dynamic_axes`. This path works with the pinned PyTorch version but is deprecated;
upgrading to the newer exporter will require validating its opset conversion and
shape behavior against the same gate. Export settings are recorded in the report.

## Train the detection baseline

`vf_ml.detection_training.train_detector` connects a frozen detection manifest to
our YOLO adapter, the pinned Ultralytics trainer, validation history, and a saved
YOLOv8s checkpoint. It always prepares a fresh train/validation layout inside the
run; test images are not opened. Rerun `scripts/setup_training_env.sh` to install
the updated pinned requirements on each machine.

For a full baseline on one available, allocated remote GPU:

```bash
cd /var/tmp/visionforge-t25089/source
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /var/tmp/visionforge-t25089/venv-cu126/bin/python -m scripts.train_detector \
  --manifest /usershome/t25089/visionforge/data/manifests/neu-det-v1/manifest.json \
  --data-dir /usershome/t25089/visionforge/data \
  --cache-dir /usershome/t25089/visionforge/models/yolo \
  --output /usershome/t25089/visionforge/runs/neu-det-baseline-1 \
  --device cuda --workers 2
```

GPU 4 is only an example; check current usage and allocation before starting.
The wrapper requires exactly one visible GPU and preserves that selection through
Ultralytics. It does not implement GPU scheduling or reservation. CPU is the CLI
default, so local development never silently selects a GPU.

Defaults are YOLOv8s pretrained on COCO, 640-pixel input, batch 16, 100 epochs, and
early-stopping patience 20. Seed 42 and deterministic mode are recorded. The pinned
Ultralytics optimizer, learning-rate schedule, and other augmentation defaults are
used; `fit/args.yaml` saves the resolved settings. Horizontal and vertical flips
are disabled unless explicitly enabled with `--horizontal-flip` / `--vertical-flip`.
CUDA uses FP16 AMP after the library's AMP check; CPU training uses float32. The run
records whether AMP was actually enabled. Batch size is explicit; automatic OOM
batch reduction and resume support are not implemented by this wrapper yet.

Why this version: **Ultralytics 8.3.221** supports YOLOv8 and avoids the newer
platform client dependency. We pin its behavior and all transitive dependencies,
then verify compatibility with our existing PyTorch 2.13 / Python 3.11 environments.
OpenCV imports and the actual image/label loader were checked on the unprivileged
VM. No system packages or sudo were needed. This is a tested version choice,
not a claim that it is the latest release.

The official `v8.3.0` release assets `yolov8s.pt` and, for CUDA's built-in AMP
check only, `yolo11n.pt` are cached with pinned byte sizes and SHA-256 checks.
Those hashes were measured from the upstream files on 2026-10-19. Modified cached
weights fail before deserialization. There is no silent fallback to random
initialization. `--random-init` is available explicitly for offline execution tests.
Ultralytics configuration is directed to the run directory before its first import;
use the CLI in a fresh process for each run. Cloud integrations and synchronization
are disabled, along with automatic package installation and online update checks.
The library may still fetch a font into its configuration directory when missing.

The loader audit checks that Ultralytics retained every prepared image and each
unique float32 label row. Repeated rows are the only permitted removal; other
missing or changed labels fail the run. On our full NEU-DET training layout, the
loader removes **one duplicate box row**: 2,909 manifest boxes become **2,908 effective
training boxes** across 1,259 images. Validation retains all 625 boxes across 270
images. The original manifest and label files are preserved, and per-run loader
counts are saved. This duplicate annotation is separate from the duplicate image
pair grouped into the test split during dataset preparation.

| Artifact | Purpose |
| --- | --- |
| `source-manifest.json` | Exact supplied snapshot, including held-out rows |
| `manifest.json` | Actual training snapshot; a recorded subset only in smoke mode |
| `dataset/` | Fresh copied images, YOLO labels, class map and provenance report |
| `run.json` | Status, configs, manifest/weight hashes, loader audit, precision, selected epoch and validation metrics |
| `history.json` | Per-epoch losses, validation metrics, fitness, learning rates, successful and skipped optimizer steps |
| `fit/args.yaml`, `fit/results.csv` | Ultralytics' resolved settings and epoch log |
| `best.pt`, `fit/weights/` | Selected checkpoint and upstream training artifacts |

Best weights and early stopping use **validation mAP@0.5:0.95**, which is fitness
in this pinned version. Tied best scores select the later epoch, matching the
trainer. The saved best checkpoint is validated once more at completion, and that
result is labelled validation in `run.json`. These are Ultralytics detection
checkpoints, not the classification checkpoint format; the classifier evaluator
and exporter cannot consume them. Detection test evaluation and ONNX export remain
separate upcoming steps.

For a quick local check:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$HOME/.local/share/visionforge/venv-cpu/bin/python" -m scripts.train_detector \
  --output runs/neu-det-cpu-smoke-20261019 --fast-dev-run
```

`--fast-dev-run` selects at most two train and one validation images per class,
sets one epoch, image size 64, batch 2, and zero loader workers. CUDA smoke mode
also starts GradScaler at 128 instead of the normal 65,536 so its few batches can
exercise updates without spending the whole run reducing an overflowing scale.
The actual initial scale is recorded; full training keeps the library default.
Both the requested
and effective settings are recorded, and `smoke_run` is true. For NEU-DET this is
12 train / 6 validation images. On 2026-10-19, the actual pretrained model completed
this check locally and on the VM CPU (`neu-det-cpu-smoke-20261019`), plus VM GPU 4
(`neu-det-gpu-smoke-20261019-v2`) with AMP enabled and its check passed. All saved a
best checkpoint and finite history. Their validation mAP is zero; these tiny,
low-resolution checks establish execution, not detector quality. The 100-epoch,
640-pixel baseline and held-out test metrics remain pending.

Existing run paths are refused. Caught failures mark `run.json` failed and keep
artifacts for diagnosis; a forcibly killed process may leave status `running`.
Non-finite training loss or metrics, or a run with zero successful optimizer
updates, fails the wrapper. Successful and overflow-skipped update counts are
recorded per epoch and per run. The initial GPU smoke run exposed this case: its
shared convolution weights did not change with the normal starting scale. The
second smoke run verified successful updates and changed checkpoint weights. No generated datasets, weights, or run files are committed.

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
