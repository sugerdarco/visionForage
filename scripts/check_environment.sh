#!/usr/bin/env bash
# Read-only inventory; failed probes are reported and do not stop later checks.
set -u

if [[ $# -gt 1 || (${1:-} != '' && ${1:-} != '--torch') ]]; then
    printf 'Usage: bash scripts/check_environment.sh [--torch]\n' >&2
    exit 2
fi

probe() {
    printf '\n>>> %s\n' "$*"
    if command -v "$1" >/dev/null 2>&1; then
        "$@"
        local status=$?
        if [[ $status -ne 0 ]]; then
            printf 'Probe failed (exit %s).\n' "$status"
        fi
    else
        printf 'Not installed or not on PATH: %s\n' "$1"
    fi
}

probe uname -srm
probe getconf _NPROCESSORS_ONLN
probe free -h
probe df -h . /tmp
probe python3 --version
probe python3.11 --version
probe uv --version
probe git --version
probe node --version
probe docker version
probe docker compose version
probe nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv

if [[ ${1:-} == '--torch' ]]; then
    probe python3 - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"Wheel CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPUs: {torch.cuda.device_count()}")
PY
fi
