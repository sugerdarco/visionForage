"""Fetch checksum-pinned upstream YOLO weights into a user-owned cache."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


WEIGHTS = {
    "yolov8s.pt": (
        22588772,
        "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a",
    ),
    "yolo11n.pt": (
        5613764,
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    ),
}
WEIGHT_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cached_weight(cache_dir: str | Path, name: str) -> Path:
    """Verify before deserialization; never replace a changed cached checkpoint."""
    size, digest = WEIGHTS[name]
    path = Path(cache_dir).resolve() / name

    def verify(candidate):
        if candidate.stat().st_size != size or file_sha256(candidate) != digest:
            raise ValueError(f"Weight size or SHA-256 mismatch: {candidate}")

    if path.exists():
        verify(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".weights-") as work:
        temporary = Path(work) / name
        with (
            urlopen(f"{WEIGHT_BASE_URL}/{name}", timeout=60) as response,
            temporary.open("wb") as target,
        ):
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > size:
                    raise ValueError("Weight download exceeds the pinned size")
                target.write(chunk)
        verify(temporary)
        # An independent downloader may have won the race; verify that file too.
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify(path)
    return path
