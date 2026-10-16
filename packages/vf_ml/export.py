"""Export a verified classifier and gate publication on CPU ONNX parity."""

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

from vf_ml.checkpoint import load_checkpoint
from vf_ml.classification import ClassificationDataset


OPSET = 17
PARITY_ATOL = 1e-3


def _check_logits(reference: np.ndarray, actual: np.ndarray, classes: int) -> float:
    if (
        reference.ndim != 2
        or reference.shape[1] != classes
        or actual.shape != reference.shape
        or actual.dtype != np.float32
        or not np.isfinite(reference).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("Model returned invalid or non-finite logits")
    error = float(np.max(np.abs(reference.astype(np.float64) - actual)))
    if error > PARITY_ATOL:
        raise ValueError(
            f"ONNX parity failed: max absolute error {error} > {PARITY_ATOL}"
        )
    return error


def export_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    split: str = "test",
    parity_samples: int = 32,
    cpu_threads: int = 2,
) -> dict:
    """Write model.onnx, metadata.json and verification.json to a new directory."""
    if type(parity_samples) is not int or parity_samples < 2:
        raise ValueError("parity_samples must be an integer of at least 2")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Export already exists: {output}")
    torch.set_num_threads(cpu_threads)
    loaded = load_checkpoint(checkpoint_path, manifest_path, split=split)
    dataset = ClassificationDataset(
        loaded.manifest, data_root, split, loaded.transforms
    )
    if len(dataset) < parity_samples:
        raise ValueError(
            f"Parity requires {parity_samples} distinct {split} images; found {len(dataset)}"
        )
    selected = [dataset[index] for index in range(parity_samples)]
    images = torch.stack([sample["image"] for sample in selected])
    classes = len(loaded.manifest.class_names)
    size = loaded.transforms.image_size
    model = loaded.model.cpu().float().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".onnx-") as work:
        work = Path(work)
        model_path = work / "model.onnx"
        with torch.inference_mode():
            # The pinned TorchScript exporter supports the plan's exact opset 17.
            # Keep this explicit: the newer exporter has different shape semantics.
            torch.onnx.export(
                model,
                (images[:1],),
                str(model_path),
                dynamo=False,
                opset_version=OPSET,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            )
        graph = onnx.load(model_path)
        onnx.checker.check_model(graph, full_check=True)
        if {entry.domain: entry.version for entry in graph.opset_import} != {"": OPSET}:
            raise ValueError("Export did not produce the required ONNX opset 17")
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != "images"
            or outputs[0].name != "logits"
            or inputs[0].type != "tensor(float)"
            or outputs[0].type != "tensor(float)"
            or inputs[0].shape != ["batch", 3, size, size]
            or outputs[0].shape != ["batch", classes]
        ):
            raise ValueError("Export has an unexpected input/output contract")
        batch_sizes = [1, min(8, parity_samples)]
        errors = []
        with torch.inference_mode():
            for batch_size in batch_sizes:
                for start in range(0, parity_samples, batch_size):
                    batch = images[start : start + batch_size]
                    reference = model(batch).numpy()
                    actual = session.run(["logits"], {"images": batch.numpy()})[0]
                    errors.append(_check_logits(reference, actual, classes))
        single = images[:1].numpy()
        for _ in range(10):
            session.run(["logits"], {"images": single})
        durations = []
        for _ in range(200):
            start = perf_counter()
            session.run(["logits"], {"images": single})
            durations.append((perf_counter() - start) * 1000)
        metadata = {
            "schema_version": 1,
            "task_type": "classification",
            "model_file": "model.onnx",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": loaded.checkpoint_hash,
            "manifest_sha256": loaded.manifest_hash,
            "checkpoint_epoch": loaded.checkpoint["epoch"],
            "selection_val_loss": loaded.checkpoint["val_loss"],
            "class_map": dict(loaded.manifest.class_map),
            "model": loaded.metadata,
            "opset": OPSET,
            "input": {
                "name": "images",
                "dtype": "float32",
                "layout": "NCHW",
                "shape": ["batch", 3, size, size],
                "preprocessing_in_graph": False,
                "preprocessing": "vf_ml.transforms.classification_transform(training=False)",
            },
            "output": {
                "name": "logits",
                "dtype": "float32",
                "shape": ["batch", classes],
                "class_names": list(loaded.manifest.class_names),
                "probabilities": "softmax along axis 1",
            },
        }
        report = {
            "schema_version": 1,
            "model_sha256": metadata["model_sha256"],
            "parity": {
                "passed": True,
                "split": split,
                "samples": parity_samples,
                "image_keys": [sample["image_key"] for sample in selected],
                "batch_sizes": batch_sizes,
                "absolute_tolerance": PARITY_ATOL,
                "relative_tolerance": 0.0,
                "max_absolute_error": max(errors),
                "meets_32_test_image_check": split == "test" and parity_samples >= 32,
            },
            "latency": {
                "scope": "ONNX Runtime session.run only; decoding and preprocessing excluded",
                "batch_size": 1,
                "warmup_runs": 10,
                "measured_runs": 200,
                "mean_ms": float(np.mean(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
            },
            "runtime": {
                "torch_version": str(torch.__version__),
                "onnx_version": onnx.__version__,
                "onnxruntime_version": ort.__version__,
                "exporter": "torchscript",
                "provider": "CPUExecutionProvider",
                "cpu_threads": cpu_threads,
                "inter_op_threads": 1,
                "platform": platform.platform(),
                "cpu_model": next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")
                    ),
                    platform.machine(),
                ),
            },
        }
        for name, contents in (
            ("metadata.json", metadata),
            ("verification.json", report),
        ):
            (work / name).write_text(
                json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        # Reserve the destination exclusively only after every check has passed.
        # verification.json is linked last and marks a completed bundle.
        output.mkdir()
        published = []
        try:
            for name in ("model.onnx", "metadata.json", "verification.json"):
                os.link(work / name, output / name)
                published.append(output / name)
        except BaseException:
            for path in published:
                path.unlink()
            output.rmdir()
            raise
    return report
