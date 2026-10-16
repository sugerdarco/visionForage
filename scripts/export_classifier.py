"""Export a classifier to ONNX after parity checks and a CPU latency measurement."""

import argparse
import json
from pathlib import Path

from vf_ml.export import export_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="Default: manifest.json beside checkpoint"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="Default: onnx/ beside checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--parity-samples", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent / "onnx"
    try:
        report = export_checkpoint(
            args.checkpoint,
            args.manifest or args.checkpoint.parent / "manifest.json",
            args.data_dir,
            output,
            split=args.split,
            parity_samples=args.parity_samples,
            cpu_threads=args.cpu_threads,
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
