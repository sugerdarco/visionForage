"""Fine-tune ResNet-18 and keep the checkpoint with the lowest validation loss."""

import argparse
from pathlib import Path

from vf_ml.training import TrainingConfig, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/casting-v2/manifest.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New run directory; existing paths are never overwritten",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--random-init", action="store_true")
    for name in ("epochs", "batch_size", "patience", "seed", "workers", "cpu_threads"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(TrainingConfig(), name),
        )
    for name in ("learning_rate", "weight_decay", "label_smoothing"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=getattr(TrainingConfig(), name),
        )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--vertical-flip", action="store_true")
    args = vars(parser.parse_args())
    manifest, data_dir, output, cache = (
        args.pop(name) for name in ("manifest", "data_dir", "output", "cache_dir")
    )
    pretrained = not args.pop("random_init")
    try:
        train_classifier(
            manifest,
            data_dir,
            output,
            TrainingConfig(**args),
            pretrained=pretrained,
            cache_dir=cache,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
