"""ML subprocess entry point: no database, broker or object-store dependencies."""

import ctypes
import json
import os
from pathlib import Path
import signal
import sys


def run(spec):
    work = Path(spec["work"])
    manifest, data, output = work / "manifest.json", work / "data", work / "run"
    common = dict(device=spec["device"], cpu_threads=spec["cpu_threads"], workers=0)
    if spec["model_arch"] == "resnet18":
        from vf_ml.training import TrainingConfig, train_classifier
        from vf_ml.evaluation import evaluate_checkpoint
        from vf_ml.export import export_checkpoint

        train_classifier(
            manifest,
            data,
            output,
            TrainingConfig(**spec["hyperparams"], **common),
            cache_dir=spec["cache"],
        )
        evaluate_checkpoint(
            output / "best.pt",
            manifest,
            data,
            work / "evaluation.json",
            device=spec["device"],
            cpu_threads=spec["cpu_threads"],
        )
        export_checkpoint(
            output / "best.pt",
            manifest,
            data,
            work / "onnx",
            parity_samples=32,
            cpu_threads=spec["cpu_threads"],
        )
    elif spec["model_arch"] == "yolov8s":
        from vf_ml.detection_training import DetectionTrainingConfig, train_detector
        from vf_ml.detection_evaluation import evaluate_detector
        from vf_ml.detection_export import export_detector

        train_detector(
            manifest,
            data,
            output,
            DetectionTrainingConfig(**spec["hyperparams"], **common),
            cache_dir=Path(spec["cache"]) / "yolo",
        )
        evaluate_detector(
            output / "best.pt",
            manifest,
            data,
            work / "evaluation.json",
            device=spec["device"],
            cpu_threads=spec["cpu_threads"],
        )
        export_detector(
            output / "best.pt",
            manifest,
            data,
            work / "onnx",
            parity_samples=32,
            cpu_threads=spec["cpu_threads"],
        )
    else:
        raise ValueError("Unsupported model architecture")


def main():
    # Linux parent-death signal closes the orphan window if the supervisor is killed.
    # Check parent identity after prctl to cover death between spawn and this call.
    expected_parent = int(sys.argv[2])
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Cannot install parent-death signal")
    if os.getppid() != expected_parent:
        raise SystemExit(1)
    run(json.loads(Path(sys.argv[1]).read_text()))


if __name__ == "__main__":
    main()
