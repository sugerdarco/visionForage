#!/usr/bin/env bash
# One-time CPU build, suitable for the remote VM without sudo or Docker.
set -euo pipefail

if [[ $# != 1 || $1 != /* ]]; then
    echo "Usage: bash scripts/build_storage_tools.sh /absolute/build/directory" >&2
    exit 1
fi
if [[ $(uname -sm) != 'Linux x86_64' ]]; then
    echo 'This pinned toolchain targets Linux x86_64.' >&2
    exit 1
fi
build_dir=$1
mkdir -p "$build_dir"/{downloads,tools,src,bin}

download() {
    local url=$1 path=$2 checksum=$3
    if [[ ! -f $path ]]; then
        curl --fail --location --retry 3 "$url" -o "$path.part"
        mv "$path.part" "$path"
    fi
    printf '%s  %s\n' "$checksum" "$path" | sha256sum --check --status
}

download https://go.dev/dl/go1.26.8.linux-amd64.tar.gz \
    "$build_dir/downloads/go.tar.gz" \
    d0f743b33e8d8945e6b1f432edd15785c70507121d6e2a723b21285eddf8b57b
download https://codeload.github.com/minio/minio/tar.gz/9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a \
    "$build_dir/downloads/minio.tar.gz" \
    45521908307306e925c98d629e1c17d78c8b72b6ee242b1bfb1409f7d8ee5841
download https://codeload.github.com/minio/mc/tar.gz/RELEASE.2025-08-13T08-35-41Z \
    "$build_dir/downloads/mc.tar.gz" \
    29db22500374169a43951c7cef09daf19e7291ea5ba00ac10f321371b0a35b32

if [[ ! -x $build_dir/tools/go/bin/go ]]; then
    tar -xzf "$build_dir/downloads/go.tar.gz" -C "$build_dir/tools"
fi
export PATH="$build_dir/tools/go/bin:$PATH"
export GOTOOLCHAIN=local GOENV=off CGO_ENABLED=0 GOOS=linux GOARCH=amd64
export GOMODCACHE="$build_dir/modules" GOCACHE="$build_dir/cache"
export GOMAXPROCS=4
[[ $(go version) == 'go version go1.26.8 linux/amd64' ]]

build_tool() {
    local name=$1 version=$2 commit=$3
    local flags="-s -w"
    flags+=" -X github.com/minio/$name/cmd.Version=$version"
    flags+=" -X github.com/minio/$name/cmd.CommitID=$commit"
    flags+=" -X github.com/minio/$name/cmd.ReleaseTag=DEVELOPMENT.${version//:/-}"
    flags+=" -X github.com/minio/$name/cmd.CopyrightYear=2025"
    mkdir -p "$build_dir/src/$name"
    tar -xzf "$build_dir/downloads/$name.tar.gz" \
        -C "$build_dir/src/$name" --strip-components=1
    (
        cd "$build_dir/src/$name"
        go build -mod=readonly -trimpath -buildvcs=false -p 4 \
            -ldflags "$flags" \
            -o "$build_dir/bin/$name" .
        cp LICENSE "$build_dir/bin/LICENSE-$name"
    )
}
build_tool minio 2025-10-15T17:29:55Z 9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a
build_tool mc 2025-08-13T08:35:41Z 7394ce0dd2a80935aded936b09fa12cbb3cb8096
(
    cd "$build_dir/bin"
    sha256sum minio mc LICENSE-minio LICENSE-mc > SHA256SUMS
    ./minio --version
    ./mc --version
)
echo "Built tools and checksums in $build_dir/bin"
