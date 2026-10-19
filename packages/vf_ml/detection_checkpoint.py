"""Verify and load trusted YOLO checkpoints with isolated library settings."""

import hashlib
import json
import os
from pathlib import Path

from vf_ml.detection_training import ULTRALYTICS_VERSION
from vf_ml.manifest import parse_manifest_json


def _sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def read_detection_run(checkpoint_path, manifest_path, run_path, split):
    """Check provenance before loading Ultralytics' pickle-based checkpoint."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    contents = Path(manifest_path).read_bytes()
    manifest = parse_manifest_json(contents.decode("utf-8"))
    run_bytes = Path(run_path).read_bytes()
    run = json.loads(run_bytes)
    if not isinstance(run, dict) or not isinstance(run.get("effective_config"), dict):
        raise ValueError("Expected a training run object with an effective_config")
    if (
        run.get("schema_version") != 1
        or run.get("status") != "completed"
        or run.get("task_type") != "detection"
        or run.get("model_name") != "yolov8s"
        or run.get("ultralytics_version") != ULTRALYTICS_VERSION
        or run.get("selection_split") != "val"
        or not isinstance(run.get("selection_metric"), str)
        or not run["selection_metric"]
    ):
        raise ValueError("Expected a completed, compatible YOLOv8s training run")
    if (
        manifest.task_type != "detection"
        or run.get("manifest_sha256") != _sha256(contents)
        or run.get("class_map") != dict(manifest.class_map)
    ):
        raise ValueError("Manifest hash or class map does not match the training run")
    if not manifest.samples_for(split):
        raise ValueError(f"Manifest has no samples for {split}")
    size = run.get("effective_config", {}).get("image_size")
    epoch = run.get("best_epoch")
    if (
        type(size) is not int
        or size < 64
        or size % 32
        or type(epoch) is not int
        or epoch < 1
        or type(run.get("smoke_run")) is not bool
    ):
        raise ValueError("Invalid training image size, best epoch or smoke flag")
    checkpoint = Path(checkpoint_path).read_bytes()
    if run.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("Checkpoint hash does not match the training run")
    return manifest, contents, run, run_bytes, checkpoint


def load_detector(checkpoint_path, class_names, config_dir):
    """Load only after read_detection_run has verified the copied checkpoint bytes."""
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO, __version__, settings

    if __version__ != ULTRALYTICS_VERSION:
        raise ValueError(
            f"Expected Ultralytics {ULTRALYTICS_VERSION}, got {__version__}"
        )
    settings.update(
        {
            "sync": False,
            **{
                key: False
                for key in (
                    "clearml",
                    "comet",
                    "dvc",
                    "hub",
                    "mlflow",
                    "neptune",
                    "raytune",
                    "tensorboard",
                    "wandb",
                )
                if key in settings
            },
        }
    )
    model = YOLO(str(checkpoint_path), task="detect")
    if model.task != "detect" or model.names != dict(enumerate(class_names)):
        raise ValueError("Checkpoint task or class order does not match the manifest")
    return model.model
