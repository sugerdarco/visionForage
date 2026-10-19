"""Export YOLOv8s raw predictions with dynamic batches and strict CPU ONNX parity."""

from copy import deepcopy
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

from vf_ml.detection_checkpoint import _sha256, load_detector, read_detection_run
from vf_ml.detection_training import ULTRALYTICS_VERSION, audit_loader
from vf_ml.yolo import prepare_yolo_dataset


OPSET = 17
PARITY_ATOL = 1e-3


def check_predictions(reference, actual, classes, anchors):
    """Check unfiltered center-xywh boxes in input pixels and per-class probabilities."""
    if (
        reference.ndim != 3
        or reference.shape[0] == 0
        or reference.dtype != np.float32
        or reference.shape[1:] != (4 + classes, anchors)
        or actual.shape != reference.shape
        or actual.dtype != np.float32
        or not np.isfinite(reference).all()
        or not np.isfinite(actual).all()
        or np.any(actual[:, 4:] < 0)
        or np.any(actual[:, 4:] > 1)
        or np.any(actual[:, 2:4] < 0)
    ):
        raise ValueError("Detector returned invalid or non-finite raw predictions")
    errors = np.abs(reference.astype(np.float64) - actual)
    maximum = float(errors.max())
    if maximum > PARITY_ATOL:
        raise ValueError(
            f"ONNX parity failed: max absolute error {maximum} > {PARITY_ATOL}"
        )
    return {"boxes": float(errors[:, :4].max()), "scores": float(errors[:, 4:].max())}


def parity_images(work, prepared, manifest, split, size, count, stride):
    """Use the same pinned square-letterbox loader and normalization as evaluation."""
    from ultralytics.models.yolo.detect import DetectionValidator

    validator = DetectionValidator(
        args={"imgsz": size, "rect": False, "half": False, "workers": 0},
        save_dir=work / "loader",
    )
    validator.data = {
        "names": dict(enumerate(manifest.class_names)),
        "nc": len(manifest.class_names),
        "channels": 3,
    }
    validator.stride = stride
    validator.device = torch.device("cpu")
    dataset = validator.build_dataset(str(work / "dataset/images" / split), batch=1)
    audit = audit_loader(dataset, prepared, work / "dataset", split)
    index_by_path = {
        str(Path(path).resolve()): index for index, path in enumerate(dataset.im_files)
    }
    record_by_key = {row["image_key"]: row for row in prepared["images"]}
    selected = manifest.samples_for(split)[:count]
    images = []
    for sample in selected:
        record = record_by_key[sample.image_key]
        path = str((work / "dataset" / record["image_file"]).resolve())
        images.append(dataset[index_by_path[path]]["img"])
    batch = validator.preprocess({"img": torch.stack(images)})["img"]
    return batch, [record_by_key[sample.image_key] for sample in selected], audit


def export_detector(
    checkpoint_path,
    manifest_path,
    data_root,
    output,
    *,
    run_path=None,
    split="test",
    parity_samples=32,
    cpu_threads=2,
):
    """Publish an ONNX bundle only after opset, shape, parity and timing checks pass."""
    if type(parity_samples) is not int or parity_samples < 2:
        raise ValueError("parity_samples must be an integer of at least 2")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Export already exists: {output}")
    manifest, contents, run, run_bytes, checkpoint = read_detection_run(
        checkpoint_path,
        manifest_path,
        run_path or Path(checkpoint_path).parent / "run.json",
        split,
    )
    if len(manifest.samples_for(split)) < parity_samples:
        raise ValueError(
            f"Parity requires {parity_samples} distinct {split} images; found {len(manifest.samples_for(split))}"
        )
    size = run["effective_config"]["image_size"]
    classes = len(manifest.class_names)
    torch.set_num_threads(cpu_threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".detection-onnx-"
    ) as directory:
        work = Path(directory).resolve()
        (work / "best.pt").write_bytes(checkpoint)
        (work / "manifest.json").write_bytes(contents)
        model = (
            load_detector(work / "best.pt", manifest.class_names, work / "settings")
            .cpu()
            .float()
            .eval()
            .fuse(verbose=False)
        )
        from ultralytics.nn.modules import C2f, Detect

        head = model.model[-1]
        if type(head) is not Detect or head.end2end or head.nc != classes or head.xyxy:
            raise ValueError(
                "Expected a standard YOLOv8 Detect head with center-xywh output"
            )
        strides = [int(value) for value in head.stride.tolist()]
        if strides != [8, 16, 32]:
            raise ValueError("Expected YOLOv8s detection strides 8, 16, 32")
        anchors = sum((size // stride) ** 2 for stride in strides)
        prepared = prepare_yolo_dataset(
            work / "manifest.json", data_root, work / "dataset", evaluation_split=split
        )
        images, records, audit = parity_images(
            work, prepared, manifest, split, size, parity_samples, max(strides)
        )
        # Use the ordinary fused inference model, as the evaluator's AutoBackend
        # does. Fuse once before copying so PyTorch and ONNX use identical weights.
        export_model = deepcopy(model).eval()
        for parameter in export_model.parameters():
            parameter.requires_grad_(False)
        for module in export_model.modules():
            if isinstance(module, Detect):
                module.export = True
                module.format = "onnx"
                module.dynamic = False  # fixed spatial grid; batch remains dynamic
                module.xyxy = False
            elif isinstance(module, C2f):
                module.forward = module.forward_split
        model_path = work / "model.onnx"
        with torch.inference_mode():
            # Populate the fixed spatial grid before tracing dynamic batch dimensions.
            preview = export_model(images[:1])
            if preview.shape != (1, 4 + classes, anchors):
                raise ValueError("Detector has an unexpected raw prediction shape")
            torch.onnx.export(
                export_model,
                (images[:1],),
                str(model_path),
                dynamo=False,
                opset_version=OPSET,
                input_names=["images"],
                output_names=["predictions"],
                dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}},
                do_constant_folding=True,
            )
        graph = onnx.load(model_path)
        # The legacy exporter leaves the Concat channel dimension symbolic.
        # Record the fixed dimensions verified above; runtime parity below checks
        # actual tensors at multiple batch sizes, not just these annotations.
        if len(graph.graph.output) != 1:
            raise ValueError("Expected one raw detection output")
        dimensions = graph.graph.output[0].type.tensor_type.shape.dim
        if len(dimensions) != 3:
            raise ValueError("Expected a three-dimensional detection output")
        for axis, expected in ((1, 4 + classes), (2, anchors)):
            if (
                dimensions[axis].HasField("dim_value")
                and dimensions[axis].dim_value != expected
            ):
                raise ValueError("Export changed a fixed output dimension")
            dimensions[axis].dim_value = expected
        onnx.checker.check_model(graph, full_check=True)
        if {entry.domain: entry.version for entry in graph.opset_import} != {"": OPSET}:
            raise ValueError("Export did not produce the required ONNX opset 17")
        onnx.save(graph, model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Additional runtime rewrites change float32 convolution arithmetic enough
        # to exceed the strict raw-pixel gate on the trained 640px detector.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != "images"
            or outputs[0].name != "predictions"
            or inputs[0].type != "tensor(float)"
            or outputs[0].type != "tensor(float)"
            or inputs[0].shape != ["batch", 3, size, size]
            or outputs[0].shape != ["batch", 4 + classes, anchors]
        ):
            raise ValueError(
                f"Export has an unexpected input/output contract: "
                f"{[(value.name, value.type, value.shape) for value in inputs + outputs]}"
            )
        batch_sizes = [1, min(8, parity_samples)]
        errors = []
        # Native CPU kernels avoid oneDNN's batch-dependent convolution rounding.
        # The checkpoint, float32 precision, raw-pixel tolerance and images stay fixed.
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            for batch_size in batch_sizes:
                for start in range(0, parity_samples, batch_size):
                    batch = images[start : start + batch_size]
                    # Reference ordinary inference, not the export-only Detect/C2f paths.
                    reference = model(batch)[0].numpy()
                    actual = session.run(["predictions"], {"images": batch.numpy()})[0]
                    errors.append(
                        check_predictions(reference, actual, classes, anchors)
                    )
        single = images[:1].numpy()
        for _ in range(10):
            session.run(["predictions"], {"images": single})
        durations = []
        for _ in range(200):
            start = perf_counter()
            session.run(["predictions"], {"images": single})
            durations.append((perf_counter() - start) * 1000)
        model_hash = _sha256(model_path.read_bytes())
        metadata = {
            "schema_version": 1,
            "task_type": "detection",
            "model_name": "yolov8s",
            "model_file": "model.onnx",
            "model_sha256": model_hash,
            "checkpoint_sha256": _sha256(checkpoint),
            "manifest_sha256": _sha256(contents),
            "run_sha256": _sha256(run_bytes),
            "checkpoint_epoch": run["best_epoch"],
            "selection_split": "val",
            "selection_metric": run["selection_metric"],
            "smoke_run": run["smoke_run"],
            "class_map": dict(manifest.class_map),
            "opset": OPSET,
            "inference": {
                "provider": "CPUExecutionProvider",
                "graph_optimization_level": "ORT_DISABLE_ALL",
                "precision": "float32",
                "pytorch_reference": "Checkpoint with standard Ultralytics Conv/BatchNorm fusion, native CPU kernels",
            },
            "input": {
                "name": "images",
                "dtype": "float32",
                "layout": "NCHW",
                "shape": ["batch", 3, size, size],
                "preprocessing_in_graph": False,
                "preprocessing": {
                    "implementation": "Ultralytics 8.3.221 DetectionValidator with rect=False",
                    "decode": "OpenCV BGR, convert to RGB before CHW",
                    "resize": "INTER_LINEAR; ratio=size/max(original_height,original_width); each resized dimension=min(ceil(original_dimension*ratio),size)",
                    "padding": "Center to square with value 114; left/top=round(total_padding/2-0.1), right/bottom=round(total_padding/2+0.1)",
                    "normalization": "float32 pixels divided by 255; no mean/std",
                },
            },
            "output": {
                "name": "predictions",
                "dtype": "float32",
                "shape": ["batch", 4 + classes, anchors],
                "boxes": "Channels 0:4 are center_x, center_y, width, height in padded input pixels",
                "scores": "Channels 4: are sigmoid class probabilities in class-map order; no objectness channel or softmax",
                "nms_in_graph": False,
                "postprocessing": "Convert xywh to xyxy, filter scores, apply class-aware NMS in input coordinates, remove padding, divide x/y by actual resize ratios, clip to original image bounds",
                "default_confidence": 0.25,
                "default_nms_iou": 0.7,
                "max_detections_per_image": 300,
                "multi_label": True,
            },
        }
        report = {
            "schema_version": 1,
            "model_sha256": model_hash,
            "smoke_run": run["smoke_run"],
            "loader_audit": audit,
            "parity": {
                "passed": True,
                "split": split,
                "samples": parity_samples,
                "image_keys": [row["image_key"] for row in records],
                "image_sha256": {
                    row["image_key"]: row["image_sha256"] for row in records
                },
                "batch_sizes": batch_sizes,
                "absolute_tolerance": PARITY_ATOL,
                "relative_tolerance": 0.0,
                "max_absolute_error": max(max(error.values()) for error in errors),
                "max_box_absolute_error": max(error["boxes"] for error in errors),
                "max_score_absolute_error": max(error["scores"] for error in errors),
                "scope": "All decoded boxes and class probabilities before confidence filtering or NMS",
                "reference": "Float32 PyTorch checkpoint with standard Ultralytics Conv/BatchNorm fusion, ordinary inference forward, native CPU kernels (oneDNN disabled)",
                "meets_32_test_image_check": split == "test" and parity_samples >= 32,
            },
            "latency": {
                "scope": "ONNX Runtime session.run only; decode, preprocessing and NMS excluded",
                "batch_size": 1,
                "warmup_runs": 10,
                "measured_runs": 200,
                "mean_ms": float(np.mean(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
            },
            "runtime": {
                "torch_version": str(torch.__version__),
                "ultralytics_version": ULTRALYTICS_VERSION,
                "onnx_version": onnx.__version__,
                "onnxruntime_version": ort.__version__,
                "exporter": "torchscript",
                "export_conv_batchnorm_fusion": True,
                "reference_conv_batchnorm_fusion": True,
                "export_constant_folding": True,
                "reference_mkldnn_enabled": False,
                "onnxruntime_graph_optimization": "ORT_DISABLE_ALL",
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
        for name, value in (("metadata.json", metadata), ("verification.json", report)):
            (work / name).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
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
