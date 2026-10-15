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
