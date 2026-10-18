"""Fine-tune YOLOv8s from a frozen detection manifest, selecting on validation only."""

import argparse
import json
from pathlib import Path

from vf_ml.detection_training import DetectionTrainingConfig, train_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/neu-det-v1/manifest.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/model-cache/yolo"))
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Offline testing only; baseline uses pretrained weights",
    )
    for name in (
        "epochs",
        "image_size",
        "batch_size",
        "patience",
        "seed",
        "workers",
        "cpu_threads",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(DetectionTrainingConfig(), name),
        )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name in ("horizontal_flip", "vertical_flip", "fast_dev_run"):
        parser.add_argument("--" + name.replace("_", "-"), action="store_true")
    args = vars(parser.parse_args())
    manifest, data, output, cache = (
        args.pop(name) for name in ("manifest", "data_dir", "output", "cache_dir")
    )
    pretrained = not args.pop("random_init")
    try:
        run = train_detector(
            manifest,
            data,
            output,
            DetectionTrainingConfig(**args),
            cache_dir=cache,
            pretrained=pretrained,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Detector training failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": run["status"],
                "best_epoch": run["best_epoch"],
                "smoke_run": run["smoke_run"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
