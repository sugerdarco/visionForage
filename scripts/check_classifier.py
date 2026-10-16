"""Check a real casting batch through ResNet-18 without running an experiment."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vf_ml.classification import ClassificationDataset
from vf_ml.manifest import load_manifest
from vf_ml.models import create_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip pretrained weights for an offline wiring check",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.device == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("Set CUDA_VISIBLE_DEVICES to one allocated GPU index or UUID")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            parser.error("Expected exactly one visible, usable CUDA GPU")
    manifest = load_manifest(args.data_dir / "manifests/casting-v2/manifest.json")
    classifier = create_classifier(
        manifest, pretrained=not args.random_init, cache_dir=args.cache_dir
    )
    dataset = ClassificationDataset(
        manifest, args.data_dir, "train", classifier.transforms
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                generator=torch.Generator().manual_seed(42),
            )
        )
    )
    model = classifier.model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.get_classifier().weight.detach().clone()
    logits = model(batch["image"].to(args.device))
    assert logits.shape == (len(batch["target"]), len(manifest.class_names))
    loss = torch.nn.functional.cross_entropy(logits, batch["target"].to(args.device))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer.step()
    assert not torch.equal(before, model.get_classifier().weight)
    print(
        json.dumps(
            {
                "checks": "forward, finite loss, backbone/head gradients, optimizer update passed",
                "device": args.device,
                "batch_shape": list(batch["image"].shape),
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "model": classifier.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
