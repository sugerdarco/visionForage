"""CPU ONNX execution and geometry checks, without downloading trained models."""

from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from bundle_fixture import make_bundle
from vf_inference.runtime import ModelRuntime, detection_input, detection_output


def picture(size=(71, 39), *, mode="RGB", color=(255, 0, 0)):
    with Image.new(mode, size, color) as image:
        stream = BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hashes = make_bundle(self.root)

    def test_real_onnx_classifier_uses_saved_transform_and_softmax(self):
        runtime = ModelRuntime(self.root, *self.hashes)
        result = runtime.predict(picture())
        self.assertEqual(result["label"], "red")
        self.assertAlmostEqual(result["confidence"], math.e / (math.e + 1), places=6)
        self.assertAlmostEqual(sum(result["scores"].values()), 1)
        self.assertEqual(runtime.session.get_providers(), ["CPUExecutionProvider"])
        grayscale = runtime.predict(picture(mode="L", color=120))
        self.assertAlmostEqual(grayscale["confidence"], 0.5)

    def test_model_and_metadata_hashes_are_required(self):
        for hashes in (("0" * 64, self.hashes[1]), (self.hashes[0], "0" * 64)):
            with self.assertRaises(ValueError):
                ModelRuntime(self.root, *hashes)
        path = self.root / "model.onnx"
        data = path.read_bytes()
        path.unlink()
        target = self.root / "elsewhere.onnx"
        target.write_bytes(data)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            ModelRuntime(self.root, *self.hashes)

    def test_mismatched_metadata_and_graph_are_rejected(self):
        import hashlib

        metadata = json.loads((self.root / "metadata.json").read_text())
        metadata["class_map"]["blue"] = 2
        metadata["output"].update(
            shape=["batch", 3], class_names=["red", "green", "blue"]
        )
        data = json.dumps(metadata).encode()
        (self.root / "metadata.json").write_bytes(data)
        with self.assertRaises(ValueError):
            ModelRuntime(self.root, self.hashes[0], hashlib.sha256(data).hexdigest())

    def test_invalid_images_and_declared_type_mismatch_are_rejected(self):
        runtime = ModelRuntime(self.root, *self.hashes)
        for data in (
            b"invalid",
            picture((9, 10)),
            picture(mode="RGBA", color=(0, 0, 0, 0)),
        ):
            with self.assertRaises(ValueError):
                runtime.predict(data)
        with self.assertRaises(ValueError):
            runtime.predict(picture(), "image/jpeg")

    def test_letterbox_preserves_aspect_and_records_actual_axis_scales(self):
        tensor, geometry = detection_input(picture((201, 101)), 640)
        width, height, sx, sy, left, top = geometry
        self.assertEqual((width, height, left, top), (201, 101, 0, 159))
        self.assertEqual((sx, sy), (640 / 201, 322 / 101))
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertTrue(np.all(tensor[0, :, 0, :] == np.float32(114) / np.float32(255)))
        self.assertTrue(np.all(tensor[0, 0, top, :] == 1))
        self.assertTrue(np.all(tensor[0, 1:, top, :] == 0))

    def test_class_aware_multilabel_nms_and_normalized_original_boxes(self):
        # 200x100 image letterboxed to 640x640, with 160 pixels of vertical padding.
        raw = np.zeros((1, 6, 3), np.float32)
        raw[0, :4, :] = np.array(
            [[320, 320, 500], [320, 320, 320], [320, 320, 20], [160, 160, 20]]
        )
        raw[0, 4:, :] = [[0.9, 0.8, 0.1], [0.85, 0.0, 0.0]]
        result = detection_output(
            raw,
            ["scratch", "dent"],
            (200, 100, 3.2, 3.2, 0, 160),
            confidence=0.25,
            iou=0.7,
            max_detections=300,
        )
        self.assertEqual([b["label"] for b in result["boxes"]], ["scratch", "dent"])
        for box in result["boxes"]:
            for key, value in {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}.items():
                self.assertAlmostEqual(box[key], value)
        self.assertEqual(
            detection_output(
                raw,
                ["scratch", "dent"],
                (200, 100, 3.2, 3.2, 0, 160),
                confidence=0.99,
                iou=0.7,
                max_detections=300,
            ),
            {"boxes": []},
        )


if __name__ == "__main__":
    unittest.main()
