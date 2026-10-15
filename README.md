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
Rerunning it synchronizes that environment to the pinned requirements, removing
packages outside the lock; use this directory only for VisionForge.

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

Run the setup guardrail tests with `python3 -m unittest discover -s tests`.
The dependency lists are generated from `requirements/torch.in`, including
transitive dependencies, separately for each backend:

```bash
uv pip compile requirements/torch.in --python python3.11 \
  --index-url https://download.pytorch.org/whl/cu126 \
  --no-annotate --no-header -o requirements/torch-cu126.txt
# Repeat with cpu in place of cu126 to update the CPU list.
```

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
splits. Duplicate and related-image checks belong in dataset preparation before
training; a random image split could overstate generalization. Inspection of all
8,648 image headers found RGB JPEGs, despite their grayscale appearance.

Downloads are verified before extraction. Incomplete downloads and failed
extractions are discarded; existing archives and extracted files are checked on
reruns. Changed files cause an error instead of being silently overwritten. The
tradeoff is a small amount of disk I/O on each rerun for reproducible inputs.

Run `make test` for the offline tests and `make lint` for Python lint and shell
syntax checks. Lint uses Ruff 0.14.1 through uv, pinned to keep checks consistent
without adding a development tool to the training environment.

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
