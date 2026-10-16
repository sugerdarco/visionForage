"""Read classification images from a frozen manifest and a local data root."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from vf_ml.manifest import Manifest
from vf_ml.transforms import ClassificationTransforms, classification_transform


class ClassificationDataset(Dataset):
    """Decode images lazily on CPU; never enumerate folders or infer class indices."""

    def __init__(
        self,
        manifest: Manifest,
        data_root: str | Path,
        split: str,
        config: ClassificationTransforms = ClassificationTransforms(),
    ) -> None:
        if manifest.task_type != "classification":
            raise ValueError("ClassificationDataset requires a classification manifest")
        self.samples = manifest.samples_for(split)
        if not self.samples:
            raise ValueError(f"Manifest has no samples for {split}")
        self.data_root = Path(data_root).resolve(strict=True)
        if not self.data_root.is_dir():
            raise ValueError("data_root must be a directory")
        self.class_names = manifest.class_names
        self.config = config
        self.transform = classification_transform(config, training=split == "train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        try:
            path = (self.data_root / sample.image_key).resolve(strict=True)
            if not path.is_relative_to(self.data_root):
                raise ValueError(f"Image escapes data_root: {sample.image_key}")
            with Image.open(path) as image:
                tensor = self.transform(image)
        except OSError as error:
            raise OSError(f"Cannot read image {sample.image_key}: {error}") from error
        return {
            "image": tensor,
            "target": sample.labels[0].class_index,
            "image_key": sample.image_key,
        }
