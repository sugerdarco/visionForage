"""Write classification metrics for a saved checkpoint and its frozen val/test split."""

import argparse
import json
from pathlib import Path

from vf_ml.evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="Default: manifest.json beside the checkpoint"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, help="Default: report.json beside the checkpoint"
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent / "report.json"
    try:
        report = evaluate_checkpoint(
            args.checkpoint,
            args.manifest or args.checkpoint.parent / "manifest.json",
            args.data_dir,
            output,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            cpu_threads=args.cpu_threads,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "split": report["split"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
