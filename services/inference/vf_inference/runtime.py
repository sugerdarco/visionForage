"""Verified ONNX loading and the exported classification/detection contracts."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from vf_inference.images import decode_image


def detection_input(data, size):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    height, width = image.shape[:2]
    ratio = size / max(height, width)
    resized_w = min(math.ceil(width * ratio), size)
    resized_h = min(math.ceil(height * ratio), size)
    if (width, height) != (resized_w, resized_h):
        image = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
    left, top = round((size - resized_w) / 2 - 0.1), round((size - resized_h) / 2 - 0.1)
    image = cv2.copyMakeBorder(
        image,
        top,
        size - resized_h - top,
        left,
        size - resized_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(
        image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
    )
    tensor /= np.float32(255)
    return tensor, (width, height, resized_w / width, resized_h / height, left, top)


def detection_output(raw, names, geometry, *, confidence, iou, max_detections):
    predictions = raw[0].T
    anchors, classes = np.nonzero(predictions[:, 4:] > confidence)
    if not len(anchors):
        return {"boxes": []}
    scores = predictions[anchors, classes + 4]
    order = np.argsort(-scores, kind="stable")[:30_000]
    anchors, classes, scores = anchors[order], classes[order], scores[order]
    centers = predictions[anchors, :2]
    half_size = predictions[anchors, 2:4] / 2
    boxes = np.concatenate((centers - half_size, centers + half_size), axis=1)
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    remaining = np.arange(len(boxes))
    keep = []
    while len(remaining) and len(keep) < max_detections:
        selected = remaining[0]
        keep.append(selected)
        rest = remaining[1:]
        intersection = np.prod(
            np.maximum(
                np.minimum(boxes[selected, 2:], boxes[rest, 2:])
                - np.maximum(boxes[selected, :2], boxes[rest, :2]),
                0,
            ),
            axis=1,
        )
        overlap = intersection / np.maximum(
            areas[selected] + areas[rest] - intersection, 1e-12
        )
        remaining = rest[(classes[rest] != classes[selected]) | (overlap <= iou)]
    width, height, scale_x, scale_y, left, top = geometry
    results = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index].astype(np.float64)
        x1, x2 = np.clip((np.array([x1, x2]) - left) / scale_x, 0, width) / width
        y1, y2 = np.clip((np.array([y1, y2]) - top) / scale_y, 0, height) / height
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(
            {
                "label": names[int(classes[index])],
                "confidence": float(scores[index]),
                "x": float(x1),
                "y": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }
        )
    return {"boxes": results}


class ModelRuntime:
    def __init__(self, directory, model_sha256, metadata_sha256, *, cpu_threads=2):
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise ValueError("CPU threads must be between one and eight")
        root = Path(directory)
        metadata_bytes = self.checked_file(
            root / "metadata.json", metadata_sha256, 1024**2
        )
        metadata = json.loads(metadata_bytes)
        model_bytes = self.checked_file(root / "model.onnx", model_sha256, 1024**3)
        if (
            metadata["model_sha256"] != model_sha256
            or metadata["model_file"] != "model.onnx"
            or metadata["schema_version"] != 1
            or metadata["opset"] != 17
        ):
            raise ValueError("Unsupported model bundle")
        mapping = metadata["class_map"]
        if (
            not isinstance(mapping, dict)
            or not 1 <= len(mapping) <= 1000
            or any(type(v) is not int for v in mapping.values())
            or sorted(mapping.values()) != list(range(len(mapping)))
        ):
            raise ValueError("Invalid class map")
        self.names = [
            name for name, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]
        self.task = metadata["task_type"]
        shape = metadata["input"]["shape"]
        if (
            len(shape) != 4
            or shape[:2] != ["batch", 3]
            or type(shape[2]) is not int
            or shape[2] != shape[3]
            or not 32 <= shape[2] <= 1280
            or metadata["input"]["name"] != "images"
            or metadata["input"]["dtype"] != "float32"
            or metadata["input"]["layout"] != "NCHW"
            or metadata["input"]["preprocessing_in_graph"] is not False
        ):
            raise ValueError("Unsupported input contract")
        self.size = shape[2]
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if self.task == "classification":
            import torch
            from vf_ml.transforms import (
                ClassificationTransforms,
                classification_transform,
            )

            config = ClassificationTransforms(**metadata["model"]["transforms"])
            if (
                config.image_size != self.size
                or metadata["model"]["evaluation_geometry"] != "full_image_resize"
                or metadata["output"]["class_names"] != self.names
            ):
                raise ValueError("Unsupported classifier preprocessing")
            torch.set_num_threads(cpu_threads)
            self.transform = classification_transform(config)
            self.output_name = "logits"
            self.output_shape = ["batch", len(self.names)]
        elif self.task == "detection":
            if self.size % 32 or self.size < 64 or metadata["model_name"] != "yolov8s":
                raise ValueError("Unsupported detector")
            inference = metadata["inference"]
            if (
                inference["provider"] != "CPUExecutionProvider"
                or inference["graph_optimization_level"] != "ORT_DISABLE_ALL"
                or inference["precision"] != "float32"
                or metadata["input"]["preprocessing"]["implementation"]
                != "Ultralytics 8.3.221 DetectionValidator with rect=False"
                or metadata["output"]["nms_in_graph"] is not False
                or metadata["output"]["multi_label"] is not True
            ):
                raise ValueError("Unsupported detector inference contract")
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            self.output_name = "predictions"
            self.output_shape = [
                "batch",
                4 + len(self.names),
                sum((self.size // stride) ** 2 for stride in (8, 16, 32)),
            ]
            self.postprocess = {
                "confidence": metadata["output"]["default_confidence"],
                "iou": metadata["output"]["default_nms_iou"],
                "max_detections": metadata["output"]["max_detections_per_image"],
            }
            if (
                any(
                    type(self.postprocess[k]) not in (int, float)
                    or not 0 <= self.postprocess[k] <= 1
                    for k in ("confidence", "iou")
                )
                or type(self.postprocess["max_detections"]) is not int
                or not 1 <= self.postprocess["max_detections"] <= 300
            ):
                raise ValueError("Invalid postprocessing limits")
        else:
            raise ValueError("Unsupported task")
        if (
            metadata["output"]["shape"] != self.output_shape
            or metadata["output"]["name"] != self.output_name
            or metadata["output"]["dtype"] != "float32"
        ):
            raise ValueError("Unsupported output contract")
        cv2.setNumThreads(cpu_threads)
        # Bytes prevent a graph from resolving external weight files from the host.
        self.session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != "images"
            or inputs[0].shape != shape
            or inputs[0].type != "tensor(float)"
            or outputs[0].name != self.output_name
            or outputs[0].shape != self.output_shape
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError("Graph differs from its metadata")

    @staticmethod
    def checked_file(path, expected, limit):
        if path.is_symlink() or not 0 < path.stat().st_size <= limit:
            raise ValueError("Invalid bundle file")
        with path.open("rb") as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Bundle file exceeds limit")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Bundle checksum mismatch")
        return data

    def predict(self, data, mime=None):
        with decode_image(data, mime) as image:
            if self.task == "classification":
                tensor = self.transform(image).numpy()[None]
            else:
                tensor, geometry = detection_input(data, self.size)
        raw = self.session.run([self.output_name], {"images": tensor})[0]
        if (
            raw.dtype != np.float32
            or list(raw.shape) != [1, *self.output_shape[1:]]
            or not np.isfinite(raw).all()
        ):
            raise RuntimeError("Model returned invalid output")
        if self.task == "classification":
            probabilities = np.exp(raw[0].astype(np.float64) - float(raw.max()))
            probabilities /= probabilities.sum()
            index = int(probabilities.argmax())
            return {
                "label": self.names[index],
                "confidence": float(probabilities[index]),
                "scores": dict(zip(self.names, map(float, probabilities))),
            }
        if (raw[:, 2:4] < 0).any() or (raw[:, 4:] < 0).any() or (raw[:, 4:] > 1).any():
            raise RuntimeError("Model returned invalid predictions")
        return detection_output(raw, self.names, geometry, **self.postprocess)
