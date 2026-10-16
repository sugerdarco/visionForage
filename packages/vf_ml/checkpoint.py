"""Load verified classification weights and their frozen input contract."""

from dataclasses import dataclass, fields
import hashlib
import io
from pathlib import Path

import timm
import torch

from vf_ml.manifest import Manifest, parse_manifest_json
from vf_ml.models import create_classifier, MODEL_NAME
from vf_ml.transforms import ClassificationTransforms


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    manifest: Manifest
    transforms: ClassificationTransforms
    metadata: dict
    checkpoint: dict
    manifest_hash: str
    checkpoint_hash: str


def load_checkpoint(
    checkpoint_path: str | Path, manifest_path: str | Path, *, split: str
) -> LoadedCheckpoint:
    """Require matching data, class order, model version, transforms and weights."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    contents = Path(manifest_path).read_bytes()
    manifest_hash = hashlib.sha256(contents).hexdigest()
    manifest = parse_manifest_json(contents.decode("utf-8"))
    if manifest.task_type != "classification" or not manifest.samples_for(split):
        raise ValueError(
            f"Expected a classification manifest with a nonempty {split} split"
        )
    checkpoint_bytes = Path(checkpoint_path).read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), weights_only=True, map_location="cpu"
    )
    del checkpoint_bytes
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("manifest_sha256") != manifest_hash
    ):
        raise ValueError("Checkpoint manifest SHA-256 does not match this snapshot")
    if not {"model_state_dict", "epoch", "val_loss"} <= checkpoint.keys():
        raise ValueError("Checkpoint is missing weights or selection metadata")
    metadata = checkpoint.get("model", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint model metadata must be an object")
    if metadata.get("class_names") != list(manifest.class_names):
        raise ValueError("Checkpoint class order does not match the manifest")
    if (
        metadata.get("model_name") != MODEL_NAME
        or metadata.get("timm_version") != timm.__version__
    ):
        raise ValueError(
            "Checkpoint model or timm version is not supported by this environment"
        )
    if metadata.get("evaluation_geometry") != "full_image_resize":
        raise ValueError("Unsupported checkpoint evaluation geometry")
    saved = metadata.get("transforms", {})
    if not isinstance(saved, dict) or set(saved) != {
        field.name for field in fields(ClassificationTransforms)
    }:
        raise ValueError("Checkpoint must contain all transform settings")
    transforms = ClassificationTransforms(**saved)
    classifier = create_classifier(manifest, transforms=transforms, pretrained=False)
    classifier.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return LoadedCheckpoint(
        classifier.model.eval(),
        manifest,
        transforms,
        metadata,
        checkpoint,
        manifest_hash,
        checkpoint_hash,
    )
