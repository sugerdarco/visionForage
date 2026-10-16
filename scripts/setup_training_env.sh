#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || $1 != /* ]]; then
    printf 'Usage: bash scripts/setup_training_env.sh /absolute/runtime/path [cpu|cu126]\n' >&2
    exit 2
fi
backend=${2:-cpu}
case "$backend" in cpu|cu126) ;; *) printf 'Unknown backend: %s\n' "$backend" >&2; exit 2 ;; esac
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf 'This setup supports Linux x86_64 only.\n' >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
requirements="$repo_dir/requirements/torch-$backend.txt"
test -f "$requirements"
umask 077
mkdir -p -- "$1"
runtime_dir=$(cd -- "$1" && pwd)
export UV_CACHE_DIR="$runtime_dir/cache"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"
export UV_NO_CONFIG=1
unset PYTHONPATH PYTHONHOME

# uv provides a prebuilt Python, avoiding system packages and compiler setup.
uv_bin="$runtime_dir/tools/bin/uv"
uv_version=$("$uv_bin" --version 2>/dev/null || true)
if [[ $uv_version != 'uv 0.12.19' && $uv_version != 'uv 0.12.19 ('* ]]; then
    download_dir=$(mktemp -d "$runtime_dir/.uv-install.XXXXXX")
    trap 'rm -rf -- "$download_dir"' EXIT
    curl --fail --location --retry 3 --connect-timeout 15 \
        https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz \
        --output "$download_dir/uv.tar.gz"
    printf '%s  %s\n' \
        '23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8' \
        "$download_dir/uv.tar.gz" | sha256sum --check
    tar -xzf "$download_dir/uv.tar.gz" -C "$download_dir"
    mkdir -p "$runtime_dir/tools/bin"
    install -m 755 "$download_dir/uv-x86_64-unknown-linux-gnu/uv" "$uv_bin"
fi
"$uv_bin" python install 3.11.16 --no-bin
venv_dir="$runtime_dir/venv-$backend"
if [[ ! -e $venv_dir ]]; then
    "$uv_bin" venv --python 3.11.16 --managed-python "$venv_dir"
fi
"$venv_dir/bin/python" -I -c 'import sys; assert sys.version_info[:3] == (3, 11, 16)'
"$uv_bin" pip sync --python "$venv_dir/bin/python" \
    --index "https://download.pytorch.org/whl/$backend" \
    --default-index https://pypi.org/simple "$requirements"
"$uv_bin" pip install --python "$venv_dir/bin/python" \
    --no-build-isolation --no-deps --editable "$repo_dir"
"$uv_bin" pip check --python "$venv_dir/bin/python"
printf '\nEnvironment ready: %s/bin/python\n' "$venv_dir"
