"""Publish reproducible dataset snapshots and refuse changed reruns."""

import hashlib
import json
from pathlib import Path
import tempfile


def publish_snapshot(output: Path, manifest: dict, report: dict) -> None:
    """Publish both files together; accept identical reruns, refuse changed snapshots."""

    def encode(value: dict) -> bytes:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

    manifest_bytes = encode(manifest)
    report = {**report, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    files = {"manifest.json": manifest_bytes, "report.json": encode(report)}
    if output.exists() or output.is_symlink():
        if (
            output.is_symlink()
            or not output.is_dir()
            or {path.name for path in output.iterdir()} != set(files)
            or any(
                (output / name).is_symlink()
                or not (output / name).is_file()
                or (output / name).read_bytes() != content
                for name, content in files.items()
            )
        ):
            raise ValueError(
                "Snapshot differs or is incomplete; choose a new --output directory"
            )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".snapshot-") as work:
        staging = Path(work) / "snapshot"
        staging.mkdir()
        for name, content in files.items():
            (staging / name).write_bytes(content)
        staging.rename(output)
