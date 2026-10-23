"""Resolve only registered job artifacts; never accept a client-supplied key."""

import hashlib
import json
import re
from uuid import UUID

from botocore.exceptions import ClientError
from fastapi import HTTPException

DOWNLOADS = {
    "weights": ("run/best.pt", "best.pt", "application/octet-stream"),
    "onnx": ("onnx/model.onnx", "model.onnx", "application/octet-stream"),
    "metadata": ("onnx/metadata.json", "metadata.json", "application/json"),
}
REPORT_LIMIT = 32 * 1024 * 1024


def unavailable():
    return HTTPException(
        status_code=409, detail="Registered model artifact unavailable"
    )


def artifact_record(row, name):
    index = row.metrics.get("artifacts", {})
    item = index.get(name) if isinstance(index, dict) else None
    if not isinstance(item, dict):
        raise unavailable()
    key, digest, size = (item.get(field) for field in ("key", "sha256", "bytes"))
    prefix = f"{row.project_id}/jobs/{row.job_id}/"
    if not isinstance(key, str) or not key.startswith(prefix):
        raise unavailable()
    attempt, separator, suffix = key[len(prefix) :].partition("/")
    try:
        canonical_attempt = str(UUID(attempt))
    except ValueError:
        raise unavailable() from None
    if (
        attempt != canonical_attempt
        or not separator
        or suffix != name
        or not isinstance(digest, str)
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
        or type(size) is not int
        or not 0 < size <= 1024**3
        or (name == "run/best.pt" and key != row.weights_key)
        or (name == "onnx/model.onnx" and key != row.onnx_key)
    ):
        raise unavailable()
    return {"key": key, "sha256": digest, "bytes": size}


def missing_object(error):
    if error.response["Error"]["Code"] in ("NoSuchKey", "NotFound", "404"):
        raise unavailable() from None
    raise error


def read_report(store, row, name):
    item = artifact_record(row, name)
    if item["bytes"] > REPORT_LIMIT:
        raise unavailable()
    try:
        data = store.read(item["key"], max_bytes=REPORT_LIMIT)
    except ClientError as error:
        missing_object(error)
    except ValueError:
        raise unavailable() from None
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise unavailable()
    try:
        report = json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        raise unavailable() from None
    if not isinstance(report, dict):
        raise unavailable()
    return report
