"""Evaluate saved classification weights without fetching pretrained models."""

import json
import os
from pathlib import Path
import tempfile

import timm
import torch
from torch.utils.data import DataLoader

from vf_ml.checkpoint import load_checkpoint
from vf_ml.classification import ClassificationDataset
from vf_ml.metrics import classification_metrics


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
    torch.set_num_threads(cpu_threads)
    loaded = load_checkpoint(checkpoint_path, manifest_path, split=split)
    manifest, transforms = loaded.manifest, loaded.transforms
    checkpoint, metadata = loaded.checkpoint, loaded.metadata
    manifest_hash, checkpoint_hash = loaded.manifest_hash, loaded.checkpoint_hash
    model = loaded.model.to(device).eval()
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
