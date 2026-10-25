"""Casting simulation: a 10% seed, then 200 uncertainty-selected vs random labels."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(rows, seed=42):
    """Stratify the initial 10%; acquisition gets only the remaining training pool."""
    rng = random.Random(seed)
    initial, pool = [], []
    for label in sorted({row["labels"][0]["class_index"] for row in rows}):
        group = sorted(
            (row for row in rows if row["labels"][0]["class_index"] == label),
            key=lambda row: row["image_key"],
        )
        rng.shuffle(group)
        count = max(1, round(len(group) * 0.1))
        initial.extend(group[:count])
        pool.extend(group[count:])
    return sorted(initial, key=lambda r: r["image_key"]), sorted(
        pool, key=lambda r: r["image_key"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from vf_ml.checkpoint import load_checkpoint
    from vf_ml.classification import ClassificationDataset
    from vf_ml.evaluation import evaluate_checkpoint
    from vf_ml.manifest import parse_manifest
    from vf_ml.training import TrainingConfig, train_classifier

    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.2)
    torch.set_num_threads(2)
    source = json.loads(args.manifest.read_text())
    parsed = parse_manifest(source)
    if parsed.task_type != "classification":
        raise ValueError("The planned experiment uses casting classification")
    audit = json.loads(args.audit.read_text())
    if audit["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Source audit does not match manifest")
    indexed = {row["image_key"]: row for row in audit["images"]}
    # Verify all bytes once before training; exact pixel groups must not cross splits.
    groups = {}
    for row in source["samples"]:
        record = indexed[row["image_key"]]
        if digest(args.data_dir / row["image_key"]) != record["sha256"]:
            raise ValueError("Prepared image differs from its audit")
        previous = groups.setdefault(record["pixel_sha256"], row["split"])
        if previous != row["split"]:
            raise ValueError("Pixel group crosses training/held-out boundaries")
    training = [r for r in source["samples"] if r["split"] == "train"]
    if len({indexed[r["image_key"]]["pixel_sha256"] for r in training}) != len(
        training
    ):
        raise ValueError("Acquisition requires unique training pixel groups")
    heldout = [r for r in source["samples"] if r["split"] != "train"]
    initial, pool = partition(training)
    if len(pool) < 200:
        raise ValueError("Acquisition needs at least 200 pool images")
    args.output.mkdir(parents=True, exist_ok=False)
    config = TrainingConfig(
        device=args.device, workers=2, horizontal_flip=True, vertical_flip=True
    )
    protocol = {
        "schema_version": 1,
        "experiment_id": args.output.name,
        "source_manifest_sha256": digest(args.manifest),
        "seed": 42,
        "initial_fraction": 0.1,
        "initial_images": len(initial),
        "pool_images": len(pool),
        "acquisition_budget": 200,
        "config": asdict(config),
        "heldout_counts": dict(Counter(r["split"] for r in heldout)),
        "label_source": "simulated oracle using existing dataset labels",
        "selection": "lowest maximum softmax probability; image-key tie break",
        "retraining": "fresh identical pretrained initialization and seed for every arm",
        "test_policy": "unchanged held-out test, evaluated only after all arms are frozen",
    }
    write_json(args.output / "protocol.json", protocol)

    def train(name, rows):
        manifest = {
            **source,
            "samples": sorted(rows + heldout, key=lambda r: r["image_key"]),
        }
        path = args.output / f"{name}.json"
        write_json(path, manifest)
        started = time.monotonic()
        result = train_classifier(
            path, args.data_dir, args.output / name, config, cache_dir=args.cache_dir
        )
        return {
            "run_id": f"{args.output.name}/{name}",
            "training_images": len(rows),
            "training_seconds": time.monotonic() - started,
            "run": result,
        }

    results = {"seed": train("seed", initial)}
    loaded = load_checkpoint(
        args.output / "seed/best.pt", args.output / "seed/manifest.json", split="val"
    )
    model = loaded.model.to(args.device).eval()
    # Pool is exposed to the inference loader as val to use deterministic transforms.
    # Its labels are never read during ranking, and no held-out examples enter it.
    pool_manifest = parse_manifest(
        {**source, "samples": [{**r, "split": "val"} for r in pool]}
    )
    dataset = ClassificationDataset(
        pool_manifest, args.data_dir, "val", loaded.transforms
    )
    ranked = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            probabilities = model(batch["image"].to(args.device)).softmax(1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Non-finite acquisition probabilities")
            for key, confidence in zip(
                batch["image_key"], probabilities.max(1).values.cpu().tolist()
            ):
                ranked.append({"image_key": key, "confidence": confidence})
    ranked.sort(key=lambda r: (r["confidence"], r["image_key"]))
    uncertain_keys = {row["image_key"] for row in ranked[:200]}
    random_keys = set(random.Random(43).sample([r["image_key"] for r in pool], 200))
    write_json(
        args.output / "acquisition.json",
        {
            "ranked_pool": ranked,
            "uncertainty_keys": sorted(uncertain_keys),
            "random_keys": sorted(random_keys),
            "overlap": len(uncertain_keys & random_keys),
            "random_acquisition_seed": 43,
        },
    )
    del model, loaded
    if args.device == "cuda":
        torch.cuda.empty_cache()
    for name, keys in (("uncertainty", uncertain_keys), ("random", random_keys)):
        results[name] = train(
            name, initial + [row for row in pool if row["image_key"] in keys]
        )
    # No test result can influence acquisition, hyperparameters or checkpoint selection.
    for name, result in results.items():
        run = args.output / name
        report = evaluate_checkpoint(
            run / "best.pt",
            run / "manifest.json",
            args.data_dir,
            run / "report.json",
            device=args.device,
        )
        result["metrics"] = report["metrics"]
        result["checkpoint_sha256"] = digest(run / "best.pt")
        result["manifest_sha256"] = digest(run / "manifest.json")
        result["report_sha256"] = digest(run / "report.json")
        result["history"] = json.loads((run / "history.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        names = {v: k for k, v in source["class_map"].items()}
        result["training_class_counts"] = dict(
            Counter(
                names[r["labels"][0]["class_index"]]
                for r in manifest["samples"]
                if r["split"] == "train"
            )
        )
    write_json(
        args.output / "results.json",
        {
            "protocol": protocol,
            "arms": results,
            "acquisition_sha256": digest(args.output / "acquisition.json"),
            "acquisition_overlap": len(uncertain_keys & random_keys),
        },
    )
    print("ACTIVE_LEARNING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
