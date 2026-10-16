"""Shared PIL-to-tensor preprocessing for classification training and inference."""

from dataclasses import dataclass
import math

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode, v2


@dataclass(frozen=True)
class ClassificationTransforms:
    """Serializable settings; record these alongside the trained model."""

    image_size: int = 224
    interpolation: str = "bilinear"
    crop_min_scale: float = 0.85
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation_degrees: float = 10.0
    brightness: float = 0.1
    contrast: float = 0.1
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if type(self.image_size) is not int or self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if self.interpolation not in ("bilinear", "bicubic"):
            raise ValueError("interpolation must be bilinear or bicubic")
        if not 0 < self.crop_min_scale <= 1:
            raise ValueError("crop_min_scale must be in (0, 1]")
        if (
            type(self.horizontal_flip) is not bool
            or type(self.vertical_flip) is not bool
        ):
            raise ValueError("Flip settings must be booleans")
        for name in ("rotation_degrees", "brightness", "contrast"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("mean", "std"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite channel values")
            if name == "std" and any(value <= 0 for value in values):
                raise ValueError("std values must be positive")
            object.__setattr__(self, name, values)


def _rgb(image: Image.Image) -> Image.Image:
    # A named function keeps the composed transform picklable for spawned workers.
    return image.convert("RGB")


def classification_transform(
    config: ClassificationTransforms, *, training: bool = False
) -> v2.Compose:
    """Return float32 CHW tensors; evaluation and serving share the default path."""
    steps = [v2.Lambda(_rgb), v2.ToImage()]
    size = (config.image_size, config.image_size)
    interpolation = InterpolationMode(config.interpolation)
    if training:
        steps.extend(
            [
                v2.RandomResizedCrop(
                    size,
                    scale=(config.crop_min_scale, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=interpolation,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5 if config.horizontal_flip else 0.0),
                v2.RandomVerticalFlip(p=0.5 if config.vertical_flip else 0.0),
                v2.RandomRotation(
                    config.rotation_degrees, interpolation=InterpolationMode.BILINEAR
                ),
                v2.ColorJitter(brightness=config.brightness, contrast=config.contrast),
            ]
        )
    else:
        # Keep the whole image: a center crop could discard defects near the rim.
        steps.append(v2.Resize(size, interpolation=interpolation, antialias=True))
    steps.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.mean, std=config.std),
            v2.ToPureTensor(),
        ]
    )
    return v2.Compose(steps)
