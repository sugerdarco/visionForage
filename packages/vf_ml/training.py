"""Train a classifier from a frozen manifest; select checkpoints on validation only."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from vf_ml.classification import ClassificationDataset
from vf_ml.manifest import load_manifest
from vf_ml.models import create_classifier


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    patience: int = 5
    seed: int = 42
    workers: int = 0
    cpu_threads: int = 2
    device: str = "cpu"
    horizontal_flip: bool = False
    vertical_flip: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if (
            type(self.horizontal_flip) is not bool
            or type(self.vertical_flip) is not bool
        ):
            raise ValueError("Flip settings must be booleans")


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def learning_rate_at(step: int, steps_per_epoch: int, config: TrainingConfig) -> float:
    """Warm up over the first epoch, then decay per batch with a cosine schedule."""
    if step < steps_per_epoch:
        return config.learning_rate * (step + 1) / steps_per_epoch
    remaining = max(1, (config.epochs - 1) * steps_per_epoch)
    progress = (step - steps_per_epoch) / remaining
    return config.learning_rate * (1 + math.cos(math.pi * progress)) / 2


def run_epoch(
    model,
    loader,
    device,
    class_names,
    *,
    optimizer=None,
    scaler=None,
    amp_dtype=None,
    config=None,
    epoch=0,
) -> dict:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    loss_sum, count, updates = 0.0, 0, 0
    for index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if training:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(
                    epoch * len(loader) + index, len(loader), config
                )
            optimizer.zero_grad(set_to_none=True)
        # Validation stays float32 and uses ordinary, unweighted cross-entropy.
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device,
                dtype=amp_dtype,
                enabled=training and amp_dtype is not None,
            ),
        ):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(
                logits,
                targets,
                label_smoothing=config.label_smoothing if training else 0.0,
            )
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss; stopping the run")
        if training:
            scaler.scale(loss).backward()
            # With fp16, GradScaler skips overflowed updates and adjusts its scale.
            if not scaler.is_enabled() and any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("Non-finite gradients; stopping the run")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(scaler.get_scale() >= previous_scale)
        size = len(targets)
        loss_sum += loss.item() * size
        count += size
        codes = (targets * len(class_names) + logits.detach().argmax(1)).cpu()
        confusion += torch.bincount(codes, minlength=len(class_names) ** 2).reshape(
            confusion.shape
        )
    if not count:
        raise ValueError("Cannot measure an empty split")
    if training and not updates:
        raise ValueError("No optimizer updates succeeded; stopping the run")
    return {
        **({"optimizer_steps": updates} if training else {}),
        "loss": loss_sum / count,
        "accuracy": confusion.diag().sum().item() / count,
        "samples": count,
        "recall": {
            name: confusion[i, i].item() / confusion[i].sum().item()
            if confusion[i].sum()
            else None
            for i, name in enumerate(class_names)
        },
    }


def _write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def train_classifier(
    manifest_path: str | Path,
    data_root: str | Path,
    output: str | Path,
    config: TrainingConfig = TrainingConfig(),
    *,
    pretrained: bool = True,
    cache_dir: str | Path | None = None,
) -> dict:
    """Write a new run directory with an exact manifest copy, metrics, and best weights."""
    if config.device == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Set CUDA_VISIBLE_DEVICES to one allocated GPU")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expected exactly one visible, usable CUDA GPU")
    output = Path(output)
    contents = Path(manifest_path).read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "status": "running",
        "config": asdict(config),
        "manifest_sha256": hashlib.sha256(contents).hexdigest(),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "sampling": "inverse_class_frequency_with_replacement",
        "deterministic_algorithms": True,
    }
    _write_json(output / "run.json", run)
    try:
        (output / "manifest.json").write_bytes(contents)
        manifest = load_manifest(output / "manifest.json")
        if manifest.task_type != "classification":
            raise ValueError("Training requires a classification manifest")
        for split in ("train", "val"):
            if not manifest.samples_for(split):
                raise ValueError(f"Manifest has no {split} samples")
        counts = [
            sum(
                sample.labels[0].class_index == index
                for sample in manifest.samples_for("train")
            )
            for index in range(len(manifest.class_names))
        ]
        if not all(counts):
            raise ValueError("Every class needs training samples")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        classifier = create_classifier(
            manifest, pretrained=pretrained, cache_dir=cache_dir
        )
        transforms = replace(
            classifier.transforms,
            horizontal_flip=config.horizontal_flip,
            vertical_flip=config.vertical_flip,
        )
        classifier.metadata["transforms"] = asdict(transforms)
        train = ClassificationDataset(manifest, data_root, "train", transforms)
        val = ClassificationDataset(manifest, data_root, "val", transforms)
        sampler = WeightedRandomSampler(
            [1 / counts[sample.labels[0].class_index] for sample in train.samples],
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        loaders = {}
        for index, (split, dataset) in enumerate((("train", train), ("val", val))):
            loaders[split] = DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler if split == "train" else None,
                num_workers=config.workers,
                pin_memory=config.device == "cuda",
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(config.seed + index),
                **({"multiprocessing_context": "spawn"} if config.workers else {}),
            )
        model = classifier.model.to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        amp_dtype = None
        if config.device == "cuda":
            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
        run.update(
            {
                "model": classifier.metadata,
                "precision": str(amp_dtype) if amp_dtype else "float32",
                "train_class_counts": dict(zip(manifest.class_names, counts)),
                "device_name": torch.cuda.get_device_name(0)
                if config.device == "cuda"
                else "cpu",
            }
        )
        _write_json(output / "run.json", run)
        history, stale, best_loss, best_epoch = [], 0, math.inf, 0
        for epoch in range(config.epochs):
            training = run_epoch(
                model,
                loaders["train"],
                config.device,
                manifest.class_names,
                optimizer=optimizer,
                scaler=scaler,
                amp_dtype=amp_dtype,
                config=config,
                epoch=epoch,
            )
            validation = run_epoch(
                model, loaders["val"], config.device, manifest.class_names
            )
            improved = validation["loss"] < best_loss
            if improved:
                best_loss, best_epoch, stale = validation["loss"], epoch + 1, 0
                checkpoint = {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model": classifier.metadata,
                    "training_config": asdict(config),
                    "manifest_sha256": run["manifest_sha256"],
                    "epoch": best_epoch,
                    "val_loss": best_loss,
                }
                torch.save(checkpoint, output / "best.tmp")
                (output / "best.tmp").replace(output / "best.pt")
            else:
                stale += 1
            row = {
                "epoch": epoch + 1,
                "train": training,
                "val": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_epoch": best_epoch,
            }
            history.append(row)
            _write_json(output / "history.json", history)
            print(json.dumps(row), flush=True)
            if stale >= config.patience:
                break
        run.update(
            {
                "status": "completed",
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "stop_reason": "early_stopping"
                if stale >= config.patience
                else "epoch_limit",
            }
        )
    except Exception as error:
        run.update({"status": "failed", "error": str(error)})
        _write_json(output / "run.json", run)
        raise
    _write_json(output / "run.json", run)
    return run
