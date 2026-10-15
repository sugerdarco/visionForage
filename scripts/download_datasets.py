"""Download a pinned dataset archive and preserve its original folder layout."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile


CATALOG = Path(__file__).resolve().parents[1] / "datasets"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["archive_bytes"]:
        raise ValueError(f"Archive size mismatch: {path}")
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {path}; expected {spec['sha256']}")


def download_archive(destination: Path, spec: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, verified download; interrupted downloads are discarded.
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".download-"
    ) as work:
        partial = Path(work) / "archive.zip"
        request = Request(spec["url"], headers={"User-Agent": "VisionForge/0.1"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > spec["archive_bytes"]:
                    raise ValueError("Download exceeds the pinned archive size")
                output.write(chunk)
        verify_archive(partial, spec)
        partial.replace(destination)


def checked_members(archive: zipfile.ZipFile, spec: dict) -> list[zipfile.ZipInfo]:
    files = []
    names = set()
    extensions = spec.get("file_extensions", (".jpg", ".jpeg", ".png"))
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = stat.S_IFMT(member.external_attr >> 16)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.filename
            or ":" in member.filename
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or member.filename in names
            or path.as_posix() != member.filename.rstrip("/")
        ):
            raise ValueError(f"Unsafe or duplicate ZIP entry: {member.filename}")
        names.add(member.filename)
        if not member.is_dir():
            if path.suffix.lower() not in extensions:
                raise ValueError(f"Unexpected dataset file: {member.filename}")
            files.append(member)
    if sum(member.file_size for member in files) != spec["unpacked_bytes"]:
        raise ValueError("Unpacked size differs from the pinned dataset")
    counts = Counter(str(PurePosixPath(member.filename).parent) for member in files)
    if dict(counts) != spec["counts"]:
        raise ValueError("File counts differ from the pinned dataset")
    return files


def verify_existing(
    archive: zipfile.ZipFile, files: list[zipfile.ZipInfo], root: Path
) -> None:
    paths = list(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Existing dataset contains a symbolic link")
    actual = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual != {member.filename for member in files}:
        raise ValueError(
            "Existing dataset has missing or unexpected files; use a new data directory"
        )
    for member in files:
        with archive.open(member) as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha256(root / member.filename) != expected:
            raise ValueError(f"Existing dataset file changed: {member.filename}")


def prepare_dataset(
    spec: dict, data_dir: Path, archive_path: Path | None = None
) -> Path:
    name = f"{spec['name']}-v{spec['version']}"
    if archive_path is None:
        archive_path = data_dir / "downloads" / f"{name}.zip"
        if not archive_path.exists():
            print(f"Downloading {name}...", flush=True)
            download_archive(archive_path, spec)
    verify_archive(archive_path, spec)
    destination = data_dir / "raw" / name
    with zipfile.ZipFile(archive_path) as archive:
        files = checked_members(archive, spec)
        if destination.exists() or destination.is_symlink():
            verify_existing(archive, files, destination)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".extract-"
        ) as work:
            staging = Path(work) / name
            staging.mkdir()
            for member in files:
                target = staging / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            staging.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("casting", "neu-det"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--archive", type=Path, help="Use an existing ZIP; its checksum must match"
    )
    args = parser.parse_args()
    spec = json.loads((CATALOG / f"{args.dataset}.json").read_text())
    try:
        destination = prepare_dataset(spec, args.data_dir, args.archive)
    except (OSError, ValueError, URLError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "path": str(destination),
                "sha256": spec["sha256"],
                "counts": spec["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
