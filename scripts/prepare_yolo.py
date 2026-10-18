"""Build YOLO images, labels and dataset.yaml from a frozen detection manifest."""

import argparse
import json
from pathlib import Path

from vf_ml.yolo import prepare_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/neu-det-v1/manifest.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing paths are refused",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also materialize held-out test data for later evaluation",
    )
    args = parser.parse_args()
    try:
        report = prepare_yolo_dataset(
            args.manifest, args.data_dir, args.output, include_test=args.include_test
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"YOLO preparation failed: {error}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": report["manifest_sha256"],
                "counts": report["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
