"""Train YOLOv8s from a frozen manifest with explicit provenance and val-only selection."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from vf_ml.detector_weights import WEIGHTS, WEIGHT_BASE_URL, cached_weight, file_sha256
from vf_ml.manifest import parse_manifest_json
from vf_ml.yolo import prepare_yolo_dataset


ULTRALYTICS_VERSION = "8.3.221"


@dataclass(frozen=True)
class DetectionTrainingConfig:
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 16
    patience: int = 20
    seed: int = 42
    workers: int = 0
    cpu_threads: int = 2
    device: str = "cpu"
    horizontal_flip: bool = False
    vertical_flip: bool = False
    fast_dev_run: bool = False

    def __post_init__(self):
        for name in ("epochs", "batch_size", "patience", "cpu_threads", "image_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size < 64 or self.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 and at least 64")
        for name in ("seed", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**32 or self.device not in ("cpu", "cuda"):
            raise ValueError("Use a seed below 2**32 and device cpu or cuda")
        for name in ("horizontal_flip", "vertical_flip", "fast_dev_run"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def smoke_manifest(contents: bytes) -> bytes:
    """Select up to two train / one val images per class without reading test images."""
    manifest = parse_manifest_json(contents.decode("utf-8"))
    selected = set()
    for split, count in (("train", 2), ("val", 1)):
        for index in range(len(manifest.class_names)):
            candidates = [
                sample
                for sample in manifest.samples_for(split)
                if any(label.class_index == index for label in sample.labels)
            ]
            selected.update(sample.image_key for sample in candidates[:count])
    data = json.loads(contents)
    data["samples"] = [
        sample for sample in data["samples"] if sample["image_key"] in selected
    ]
    return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def audit_loader(dataset, prepared: dict, root: Path, split: str) -> dict:
    """Allow only Ultralytics' float32 duplicate-row removal; reject silent data loss."""
    expected = {
        str((root / row["image_file"]).resolve()): row
        for row in prepared["images"]
        if row["split"] == split
    }
    actual = {str(Path(row["im_file"]).resolve()): row for row in dataset.labels}
    if len(actual) != len(dataset.labels) or actual.keys() != expected.keys():
        raise ValueError(f"Ultralytics changed the {split} image set")
    original_boxes = effective_boxes = 0
    for key, record in expected.items():
        lines = (root / record["label_file"]).read_text().splitlines()
        wanted = np.array([line.split() for line in lines], dtype=np.float32).reshape(
            -1, 5
        )
        unique = np.unique(wanted, axis=0)
        loaded = actual[key]
        got = np.concatenate((loaded["cls"], loaded["bboxes"]), axis=1)
        if (
            not loaded["normalized"]
            or loaded["bbox_format"] != "xywh"
            or len(got) != len(unique)
            or not np.array_equal(np.unique(got, axis=0), unique)
        ):
            raise ValueError(f"Ultralytics changed labels for {record['image_key']}")
        original_boxes += len(wanted)
        effective_boxes += len(got)
    return {
        "images": len(actual),
        "manifest_boxes": original_boxes,
        "effective_boxes": effective_boxes,
        "duplicate_rows_removed": original_boxes - effective_boxes,
    }


def train_detector(
    manifest_path,
    data_root,
    output,
    config=DetectionTrainingConfig(),
    *,
    cache_dir,
    pretrained=True,
) -> dict:
    """Create one new run; classification and detection checkpoints are separate formats."""
    if config.device == "cuda" and (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError(
            "Select exactly one available allocated GPU with CUDA_VISIBLE_DEVICES"
        )
    source = Path(manifest_path).read_bytes()
    manifest = parse_manifest_json(source.decode("utf-8"))
    if manifest.task_type != "detection":
        raise ValueError("Detector training requires a detection manifest")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Run already exists: {output}")
    effective = (
        replace(config, epochs=1, image_size=64, batch_size=2, workers=0)
        if config.fast_dev_run
        else config
    )
    snapshot = smoke_manifest(source) if config.fast_dev_run else source
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "schema_version": 1,
        "status": "running",
        "task_type": "detection",
        "model_name": "yolov8s",
        "config": asdict(config),
        "effective_config": asdict(effective),
        "source_manifest_sha256": hashlib.sha256(source).hexdigest(),
        "manifest_sha256": hashlib.sha256(snapshot).hexdigest(),
        "class_map": dict(manifest.class_map),
        "pretrained": pretrained,
        "selection_split": "val",
        "selection_metric": "validation mAP@0.5:0.95 (Ultralytics fitness)",
        "smoke_run": config.fast_dev_run,
        "torch_version": str(torch.__version__),
    }
    _write_json(output / "run.json", run)
    try:
        (output / "source-manifest.json").write_bytes(source)
        (output / "manifest.json").write_bytes(snapshot)
        prepared = prepare_yolo_dataset(
            output / "manifest.json", data_root, output / "dataset"
        )
        weights = cached_weight(cache_dir, "yolov8s.pt") if pretrained else None
        if effective.device == "cuda":
            cached_weight(cache_dir, "yolo11n.pt")
        run["weights"] = {
            name: {"url": f"{WEIGHT_BASE_URL}/{name}", "sha256": WEIGHTS[name][1]}
            for name in (
                (["yolov8s.pt"] if pretrained else [])
                + (["yolo11n.pt"] if effective.device == "cuda" else [])
            )
        }
        # Set before the first Ultralytics import in this CLI process.
        os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics-settings")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import __version__, settings
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.utils import callbacks

        if __version__ != ULTRALYTICS_VERSION:
            raise ValueError(
                f"Expected Ultralytics {ULTRALYTICS_VERSION}, got {__version__}"
            )
        settings.update(
            {
                "sync": False,
                "weights_dir": str(Path(cache_dir).resolve()),
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

        class TrackedTrainer(DetectionTrainer):
            optimizer_steps = 0
            skipped_optimizer_steps = 0

            def optimizer_step(self):
                previous_scale = self.scaler.get_scale()
                super().optimizer_step()
                if self.scaler.get_scale() < previous_scale:
                    self.skipped_optimizer_steps += 1
                else:
                    self.optimizer_steps += 1

        history = []
        callback_map = callbacks.get_default_callbacks()

        def ready(trainer):
            if config.fast_dev_run and trainer.amp:
                # A six-batch smoke run can finish before the default 65536 scale
                # settles. Use a conservative starting scale only for this check.
                trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
            run["amp_initial_scale"] = trainer.scaler.get_scale()
            run["loader_audit"] = {
                "train": audit_loader(
                    trainer.train_loader.dataset, prepared, output / "dataset", "train"
                ),
                "val": audit_loader(
                    trainer.test_loader.dataset, prepared, output / "dataset", "val"
                ),
            }
            run["amp_enabled"] = bool(trainer.amp)
            run["device_name"] = (
                torch.cuda.get_device_name(0) if effective.device == "cuda" else "cpu"
            )
            _write_json(output / "run.json", run)

        def batch_end(trainer):
            if not torch.isfinite(trainer.loss).all():
                raise ValueError("Non-finite detector training loss")

        def epoch_end(trainer):
            epoch = trainer.epoch + 1
            if history and history[-1]["epoch"] == epoch:
                return  # final best-checkpoint validation calls this callback again
            metrics = {key: float(value) for key, value in trainer.metrics.items()}
            values = {
                **trainer.label_loss_items(trainer.tloss, prefix="train"),
                **metrics,
                **trainer.lr,
            }
            if not all(
                math.isfinite(float(value)) for value in values.values()
            ) or not math.isfinite(float(trainer.fitness)):
                raise ValueError("Non-finite detector epoch metrics")
            history.append(
                {
                    "epoch": epoch,
                    "fitness": float(trainer.fitness),
                    "optimizer_steps": trainer.optimizer_steps
                    - sum(row["optimizer_steps"] for row in history),
                    "skipped_optimizer_steps": trainer.skipped_optimizer_steps
                    - sum(row["skipped_optimizer_steps"] for row in history),
                    "metrics": {key: float(value) for key, value in values.items()},
                }
            )
            _write_json(output / "history.json", history)

        callback_map["on_pretrain_routine_end"].append(ready)
        callback_map["on_train_batch_end"].append(batch_end)
        callback_map["on_fit_epoch_end"].append(epoch_end)
        overrides = dict(
            model=str(weights) if weights else "yolov8s.yaml",
            data=str(output / "dataset/dataset.yaml"),
            epochs=effective.epochs,
            imgsz=effective.image_size,
            batch=effective.batch_size,
            patience=effective.patience,
            seed=effective.seed,
            workers=effective.workers,
            device=os.environ["CUDA_VISIBLE_DEVICES"]
            if effective.device == "cuda"
            else "cpu",
            amp=effective.device == "cuda",
            deterministic=True,
            project=str(output),
            name="fit",
            exist_ok=False,
            save=True,
            plots=False,
            val=True,
            split="val",
            resume=False,
            fliplr=0.5 if effective.horizontal_flip else 0.0,
            flipud=0.5 if effective.vertical_flip else 0.0,
        )
        if config.fast_dev_run:
            overrides["close_mosaic"] = 0
        trainer = TrackedTrainer(overrides=overrides, _callbacks=callback_map)
        torch.set_num_threads(effective.cpu_threads)
        trainer.train()
        if not history or not trainer.best.is_file():
            raise ValueError("Training did not produce history and a best checkpoint")
        if not trainer.optimizer_steps:
            raise ValueError(
                "No optimizer updates succeeded; refusing a successful run"
            )
        shutil.copyfile(trainer.best, output / "best.pt")
        best = max(history, key=lambda row: (row["fitness"], row["epoch"]))
        final_metrics = {key: float(value) for key, value in trainer.metrics.items()}
        if not all(math.isfinite(value) for value in final_metrics.values()):
            raise ValueError("Non-finite final validation metrics")
        run.update(
            status="completed",
            ultralytics_version=__version__,
            best_epoch=best["epoch"],
            best_fitness=best["fitness"],
            epochs_completed=len(history),
            optimizer_steps=trainer.optimizer_steps,
            skipped_optimizer_steps=trainer.skipped_optimizer_steps,
            checkpoint_sha256=file_sha256(output / "best.pt"),
            validation_metrics=final_metrics,
        )
        _write_json(output / "run.json", run)
    except BaseException as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(output / "run.json", run)
        raise
    return run
