"""Evaluate a trusted detector run against its exact frozen validation/test split."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from vf_ml.detection_training import ULTRALYTICS_VERSION, audit_loader
from vf_ml.detection_checkpoint import _sha256, load_detector, read_detection_run
from vf_ml.yolo import prepare_yolo_dataset


AP_CONFIDENCE = 0.001
DEFAULT_CONFIDENCE = 0.25
NMS_IOU = 0.7
MAX_DETECTIONS = 300


def _validator_class():
    # Lazy import: the CLI sets the isolated library configuration first.
    from ultralytics.models.yolo.detect import DetectionValidator

    class ReportValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.fixed_counts = np.zeros((self.nc, 3), dtype=np.int64)  # TP, FP, FN

        def postprocess(self, predictions):
            def check_finite(value):
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError("Detector returned non-finite predictions")
                if isinstance(value, (tuple, list)):
                    for item in value:
                        check_finite(item)

            check_finite(predictions)
            return super().postprocess(predictions)

        def update_metrics(self, predictions, batch):
            super().update_metrics(predictions, batch)
            for index, pred in enumerate(predictions):
                target = self._prepare_batch(index, batch)
                keep = pred["conf"] >= DEFAULT_CONFIDENCE
                filtered = {key: value[keep] for key, value in pred.items()}
                # Rematch after filtering: a discarded low-confidence box must not
                # consume the target of a retained, slightly lower-IoU prediction.
                matched = self._process_batch(filtered, target)["tp"][:, 0]
                predicted_classes = filtered["cls"].cpu().numpy().astype(int)
                target_classes = target["cls"].cpu().numpy().astype(int)
                tp = np.bincount(predicted_classes[matched], minlength=self.nc)
                predicted = np.bincount(predicted_classes, minlength=self.nc)
                support = np.bincount(target_classes, minlength=self.nc)
                self.fixed_counts += np.stack(
                    (tp, predicted - tp, support - tp), axis=1
                )

    return ReportValidator


def detection_metrics(validator, class_names):
    """Use AP only for classes with targets; report fixed-threshold counts for all."""
    box = validator.metrics.box
    ap_by_class = {
        int(index): (float(box.ap50[row]), float(box.ap[row]))
        for row, index in enumerate(box.ap_class_index)
    }

    def counts(tp, fp, fn):
        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "support": int(tp + fn),
            "predictions": int(tp + fp),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        }

    per_class = []
    for index, name in enumerate(class_names):
        ap50, ap = ap_by_class.get(index, (None, None))
        per_class.append(
            {
                "class_index": index,
                "class_name": name,
                "ap50": ap50,
                "ap50_95": ap,
                **counts(*validator.fixed_counts[index]),
            }
        )
    return {
        "map50": float(np.mean([ap[0] for ap in ap_by_class.values()]))
        if ap_by_class
        else None,
        "map50_95": float(np.mean([ap[1] for ap in ap_by_class.values()]))
        if ap_by_class
        else None,
        "fixed_threshold_micro": counts(*validator.fixed_counts.sum(axis=0)),
        "per_class": per_class,
    }


def evaluate_detector(
    checkpoint_path,
    manifest_path,
    data_root,
    output,
    *,
    run_path=None,
    split="test",
    batch_size=16,
    device="cpu",
    cpu_threads=2,
):
    """Evaluate in float32 and atomically publish one new JSON report."""
    if any(type(n) is not int or n < 1 for n in (batch_size, cpu_threads)):
        raise ValueError("batch_size and cpu_threads must be positive integers")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError(
            "Select exactly one available allocated GPU with CUDA_VISIBLE_DEVICES"
        )
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Report already exists: {output}")
    manifest, contents, run, run_bytes, checkpoint = read_detection_run(
        checkpoint_path,
        manifest_path,
        run_path or Path(checkpoint_path).parent / "run.json",
        split,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".detection-evaluation-"
    ) as work:
        work = Path(work).resolve()
        # Evaluate exactly the bytes we hashed, even if original files change later.
        (work / "manifest.json").write_bytes(contents)
        (work / "best.pt").write_bytes(checkpoint)
        prepared = prepare_yolo_dataset(
            work / "manifest.json",
            data_root,
            work / "dataset",
            evaluation_split=split,
        )
        model = load_detector(work / "best.pt", manifest.class_names, work / "settings")
        from ultralytics.utils import callbacks

        loader_audit = {}

        def audit(validator):
            torch.set_num_threads(cpu_threads)
            loader_audit.update(
                audit_loader(
                    validator.dataloader.dataset,
                    prepared,
                    work / "dataset",
                    split,
                )
            )

        callback_map = callbacks.get_default_callbacks()
        callback_map["on_val_start"].append(audit)
        torch.set_num_threads(cpu_threads)
        validator = _validator_class()(
            args={
                "task": "detect",
                "mode": "val",
                "data": str(work / "dataset/dataset.yaml"),
                "split": split,
                "imgsz": run["effective_config"]["image_size"],
                "batch": batch_size,
                "workers": 0,
                "half": False,
                "rect": False,
                "device": os.environ["CUDA_VISIBLE_DEVICES"]
                if device == "cuda"
                else "cpu",
                "conf": AP_CONFIDENCE,
                "iou": NMS_IOU,
                "max_det": MAX_DETECTIONS,
                "plots": False,
                "save_json": False,
                "verbose": False,
            },
            save_dir=work / "validation",
            _callbacks=callback_map,
        )
        validator(model=model)
        metrics = detection_metrics(validator, manifest.class_names)
        if (
            metrics["fixed_threshold_micro"]["support"]
            != loader_audit["effective_boxes"]
        ):
            raise ValueError(
                "Evaluation target count does not match the audited loader"
            )
        report = {
            "schema_version": 1,
            "task_type": "detection",
            "model_name": "yolov8s",
            "split": split,
            "smoke_run": run["smoke_run"],
            "manifest_sha256": _sha256(contents),
            "checkpoint_sha256": _sha256(checkpoint),
            "run_sha256": _sha256(run_bytes),
            "checkpoint_epoch": run["best_epoch"],
            "selection_split": "val",
            "selection_metric": run["selection_metric"],
            "class_map": dict(manifest.class_map),
            "loader_audit": loader_audit,
            "metrics": metrics,
            "metric_settings": {
                "ap_confidence_floor": AP_CONFIDENCE,
                "ap_iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
                "precision_recall_confidence": DEFAULT_CONFIDENCE,
                "precision_recall_iou": 0.5,
                "nms_iou": NMS_IOU,
                "max_detections_per_image": MAX_DETECTIONS,
                "matching": "Ultralytics class-aware greedy IoU matching, rematched at fixed confidence",
                "absent_class_ap": None,
                "zero_denominator_precision_recall": 0.0,
                "map_average": "Classes with ground-truth boxes only; null when no targets",
            },
            "runtime": {
                "torch_version": str(torch.__version__),
                "ultralytics_version": ULTRALYTICS_VERSION,
                "device": device,
                "precision": "float32",
                "batch_size": batch_size,
                "cpu_threads": cpu_threads,
                "image_size": run["effective_config"]["image_size"],
                "preprocessing": "Ultralytics square letterbox, RGB, NCHW, divide by 255",
            },
        }
        temporary = work / "report.json"
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.link(temporary, output)
    return report
