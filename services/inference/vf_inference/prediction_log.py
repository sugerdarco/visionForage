"""Durable local outbox; the CPU worker owns database and object-storage access."""

from datetime import datetime, timezone
import hashlib
import json
import os

MAX_PENDING = 1000
MAX_PENDING_BYTES = 1024**3


def enqueue(settings, identifier, data, mime, output, latency_ms):
    root = settings.prediction_log_dir
    if root is None:
        return
    sizes = []
    for path in root.glob("*.image"):
        try:
            sizes.append(path.stat().st_size)
        except FileNotFoundError:
            pass  # The collector may finish an event during this scan.
    if len(sizes) >= MAX_PENDING or sum(sizes) + len(data) > MAX_PENDING_BYTES:
        raise OSError("Prediction outbox full")
    image = root / f"{identifier}.image"
    temporary = root / f".{identifier}.tmp"
    event = root / f"{identifier}.json"
    payload = {
        "schema_version": 1,
        "prediction_id": str(identifier),
        "deployment_id": str(settings.deployment_id),
        "model_id": str(settings.model_id),
        "model_version": settings.model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "mime": mime,
        "image_bytes": len(data),
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "output": output,
    }
    encoded = json.dumps(payload, allow_nan=False).encode()
    if len(encoded) > 256 * 1024:
        raise ValueError("Prediction event too large")
    try:
        for path, content in ((image, data), (temporary, encoded)):
            with path.open("xb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        temporary.rename(event)  # Only final JSON files are visible to the collector.
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if not event.exists():
            image.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
        raise
