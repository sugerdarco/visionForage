"""Materialize verified version artifacts without querying live labels or images."""

import hashlib
import json
from pathlib import Path, PurePosixPath

from vf_api.image_files import decode_image
from vf_ml.manifest import parse_manifest_json

MAX_JSON = 32 * 1024**2


def checked_json(store, key, digest):
    data = store.read(key, max_bytes=MAX_JSON)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Frozen document checksum mismatch")
    return data


def materialize(version, project_id, task, work: Path, datasets, images, check):
    prefix = f"{project_id}/v{version.version}/"
    key = version.manifest_key
    audit_key = version.counts["audit_key"]
    if (
        not key.startswith(prefix)
        or PurePosixPath(key).name != "manifest.json"
        or str(PurePosixPath(key)) != key
        or ".." in PurePosixPath(key).parts
        or audit_key != str(PurePosixPath(key).with_name("audit.json"))
    ):
        raise ValueError("Invalid frozen artifact namespace")
    check()
    contents = checked_json(datasets, key, version.counts["manifest_sha256"])
    manifest = parse_manifest_json(contents.decode("utf-8"))
    audit_bytes = checked_json(datasets, audit_key, version.counts["audit_sha256"])
    audit = json.loads(audit_bytes)
    class_map = dict(manifest.class_map)
    if (
        manifest.task_type != task
        or audit["schema_version"] != 1
        or audit["task_type"] != task
        or audit["class_map"] != class_map
        or version.counts["class_map"] != class_map
        or audit["counts"]["manifest_sha256"] != version.counts["manifest_sha256"]
        or not 1 <= len(manifest.samples) <= 10000
        or len(class_map) > 1000
        or (task == "classification" and len(class_map) < 2)
    ):
        raise ValueError("Frozen dataset metadata disagrees")
    for split, minimum in (("train", 1), ("val", 1), ("test", 32)):
        count = len(manifest.samples_for(split))
        if (
            count < minimum
            or version.counts["images"][split] != count
            or audit["counts"]["images"][split] != count
        ):
            raise ValueError("Frozen split count is invalid")
    records = {row["image_key"]: row for row in audit["images"]}
    if len(records) != len(audit["images"]) or set(records) != {
        sample.image_key for sample in manifest.samples
    }:
        raise ValueError("Frozen audit image set disagrees")
    root = work / "data"
    root.mkdir(mode=0o700)
    received = pixels = 0
    pixel_splits = {}
    for sample in manifest.samples:
        check()
        if not sample.image_key.startswith(f"{project_id}/originals/"):
            raise ValueError("Image belongs to another project namespace")
        row = records[sample.image_key]
        if row["split"] != sample.split:
            raise ValueError("Frozen split assignment disagrees")
        data = images.read(sample.image_key)
        received += len(data)
        if received > 1024**3 or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Original bytes exceed limits or fail checksum")
        with decode_image(data) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Original dimensions disagree")
            pixels += image.width * image.height
            if pixels > 1_000_000_000:
                raise ValueError("Decoded pixel limit exceeded")
            with image.convert("RGB") as rgb:
                digest = hashlib.sha256(
                    f"RGB:{rgb.width}:{rgb.height}\0".encode("ascii") + rgb.tobytes()
                ).hexdigest()
        if (
            digest != row["pixel_sha256"]
            or pixel_splits.setdefault(digest, sample.split) != sample.split
        ):
            raise ValueError("Original pixels disagree or leak between splits")
        target = root / sample.image_key
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
    (work / "manifest.json").write_bytes(contents)
    (work / "audit.json").write_bytes(audit_bytes)
    return manifest
