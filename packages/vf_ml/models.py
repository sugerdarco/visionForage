"""Build the classification baseline with explicit weight and input provenance."""

from dataclasses import asdict, dataclass
from pathlib import Path

import timm
from timm.data import resolve_data_config
from torch import nn

from vf_ml.manifest import Manifest
from vf_ml.transforms import ClassificationTransforms


MODEL_NAME = "resnet18.a1_in1k"
WEIGHT_REPOSITORY = "timm/resnet18.a1_in1k"
WEIGHT_REVISION = "491b427b45c94c7fb0e78b5474cc919aff584bbf"
WEIGHT_FILENAME = "model.safetensors"


@dataclass
class Classifier:
    model: nn.Module
    transforms: ClassificationTransforms
    metadata: dict


def create_classifier(
    manifest: Manifest,
    *,
    transforms: ClassificationTransforms | None = None,
    pretrained: bool = True,
    cache_dir: str | Path | None = None,
) -> Classifier:
    """Return a CPU model with one logit per class; download errors are not hidden."""
    if manifest.task_type != "classification" or len(manifest.class_names) < 2:
        raise ValueError(
            "Classifier requires a classification manifest with at least two classes"
        )
    source_config = timm.models.get_pretrained_cfg(MODEL_NAME).to_dict()
    data_config = resolve_data_config(source_config)
    expected = ClassificationTransforms(
        image_size=data_config["input_size"][-1],
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
    )
    transforms = transforms or expected
    for field in ("image_size", "interpolation", "mean", "std"):
        if getattr(transforms, field) != getattr(expected, field):
            raise ValueError(
                f"{field} does not match {MODEL_NAME}: expected {getattr(expected, field)}"
            )
    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=len(manifest.class_names),
        pretrained_cfg_overlay={
            "hf_hub_id": f"{WEIGHT_REPOSITORY}@{WEIGHT_REVISION}",
            "hf_hub_filename": WEIGHT_FILENAME,
            "url": "",
        },
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    metadata = {
        "model_name": MODEL_NAME,
        "timm_version": timm.__version__,
        "class_names": list(manifest.class_names),
        "pretrained": pretrained,
        "weights": {
            "repository": WEIGHT_REPOSITORY,
            "revision": WEIGHT_REVISION,
            "filename": WEIGHT_FILENAME,
        }
        if pretrained
        else None,
        "transforms": asdict(transforms),
        "source_data_config": data_config,
        "evaluation_geometry": "full_image_resize",
    }
    return Classifier(model, transforms, metadata)
