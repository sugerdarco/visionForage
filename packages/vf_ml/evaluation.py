"""Evaluate saved classification weights without fetching pretrained models."""

from dataclasses import fields
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import timm
import torch
from torch.utils.data import DataLoader

from vf_ml.classification import ClassificationDataset
from vf_ml.manifest import parse_manifest_json
from vf_ml.metrics import classification_metrics
from vf_ml.models import MODEL_NAME, create_classifier
from vf_ml.transforms import ClassificationTransforms


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    split: str = "test",
    batch_size: int = 64,
    device: str = "cpu",
    cpu_threads: int = 2,
) -> dict:
    """Score one frozen split in float32 and publish a new report without overwriting."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(cpu_threads) is not int
        or cpu_threads < 1
    ):
        raise ValueError("batch_size and cpu_threads must be positive integers")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError(
            "Select exactly one available allocated GPU with CUDA_VISIBLE_DEVICES"
        )
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report already exists: {output}")
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
    torch.set_num_threads(cpu_threads)
    classifier = create_classifier(manifest, transforms=transforms, pretrained=False)
    classifier.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = classifier.model.to(device).eval()
    dataset = ClassificationDataset(manifest, data_root, split, transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    targets, probabilities, keys, total_loss = [], [], [], 0.0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device))
            if (
                logits.shape != (len(batch["target"]), len(manifest.class_names))
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("Model returned invalid or non-finite logits")
            total_loss += torch.nn.functional.cross_entropy(
                logits.double(), batch["target"].to(device), reduction="sum"
            ).item()
            probabilities.append(logits.softmax(1).cpu())
            targets.append(batch["target"])
            keys.extend(batch["image_key"])
    metrics = classification_metrics(
        torch.cat(targets).numpy(),
        torch.cat(probabilities).numpy(),
        keys,
        manifest.class_names,
    )
    metrics["loss"] = total_loss / len(dataset)
    report = {
        "schema_version": 1,
        "task_type": "classification",
        "split": split,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"],
        "selection_val_loss": checkpoint["val_loss"],
        "model": metadata,
        "metrics": metrics,
        "runtime": {
            "torch_version": str(torch.__version__),
            "timm_version": timm.__version__,
            "device": device,
            "precision": "float32",
            "loss_precision": "float64",
            "batch_size": batch_size,
            "cpu_threads": cpu_threads,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".evaluation-") as work:
        temporary = Path(work) / "report.json"
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # Link the completed file atomically; an existing destination always fails.
        os.link(temporary, output)
    return report
