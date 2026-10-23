"""Tiny ONNX graph of RGB channel means, not a trained model or checkpoint."""

import base64
import hashlib
import json
from pathlib import Path

# Opset 17: ReduceMean over H/W, then select red/green channels through Gemm.
GRAPH = base64.b64decode(
    "CAg6/AEKOgoGaW1hZ2VzEgRtZWFuIgpSZWR1Y2VNZWFuKg0KBGF4ZXNAAkADoAEHKg8KCGtlZXBkaW1zGACgAQIKIwoEbWVhbgoHd2VpZ2h0cwoEYmlhcxIGbG9naXRzIgRHZW1tEhJ0ZXN0LWNoYW5uZWwtbWVhbnMqKQgDCAIQAUIHd2VpZ2h0c0oYAACAPwAAAAAAAAAAAACAPwAAAAAAAAAAKhQIAhABQgRiaWFzSggAAAAAAAAAAFolCgZpbWFnZXMSGwoZCAESFQoHEgViYXRjaAoCCAMKAgggCgIIIGIdCgZsb2dpdHMSEwoRCAESDQoHEgViYXRjaAoCCAJCBAoAEBE="
)


def make_bundle(directory):
    root = Path(directory)
    model_hash = hashlib.sha256(GRAPH).hexdigest()
    metadata = {
        "schema_version": 1,
        "task_type": "classification",
        "model_file": "model.onnx",
        "model_sha256": model_hash,
        "opset": 17,
        "class_map": {"red": 0, "green": 1},
        "model": {
            "evaluation_geometry": "full_image_resize",
            "transforms": {
                "image_size": 32,
                "interpolation": "bicubic",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        },
        "input": {
            "name": "images",
            "dtype": "float32",
            "layout": "NCHW",
            "shape": ["batch", 3, 32, 32],
            "preprocessing_in_graph": False,
        },
        "output": {
            "name": "logits",
            "dtype": "float32",
            "shape": ["batch", 2],
            "class_names": ["red", "green"],
        },
    }
    data = json.dumps(metadata).encode()
    (root / "model.onnx").write_bytes(GRAPH)
    (root / "metadata.json").write_bytes(data)
    return model_hash, hashlib.sha256(data).hexdigest()
