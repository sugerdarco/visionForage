"""Validate and load the dataset snapshot consumed by training workers."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Label:
    class_index: int
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class Sample:
    image_key: str
    split: str
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    task_type: str
    class_names: tuple[str, ...]
    samples: tuple[Sample, ...]

    @property
    def class_map(self) -> Mapping[str, int]:
        return MappingProxyType(
            {name: index for index, name in enumerate(self.class_names)}
        )

    def samples_for(self, split: str) -> tuple[Sample, ...]:
        """Select a split without changing the snapshot or its class indices."""
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return tuple(sample for sample in self.samples if sample.split == split)


def _object(value: object, fields: set[str], location: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{location}: expected fields {sorted(fields)}")
    return value


def _label(value: object, task: str, class_count: int, location: str) -> Label:
    data = _object(value, {"class_index", "bbox"}, location)
    index, box = data["class_index"], data["bbox"]
    if type(index) is not int or not 0 <= index < class_count:
        raise ValueError(f"{location}: class_index is not in class_map")
    if task == "classification":
        if box is not None:
            raise ValueError(f"{location}: classification bbox must be null")
        return Label(index, None)
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{location}: detection bbox must be [x, y, width, height]")
    if any(type(number) not in (int, float) or not 0 <= number <= 1 for number in box):
        raise ValueError(f"{location}: bbox values must be finite numbers in [0, 1]")
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
        raise ValueError(
            f"{location}: bbox must have positive area and fit inside the image"
        )
    return Label(index, (x, y, width, height))


def parse_manifest(value: object) -> Manifest:
    """Validate decoded JSON and copy it into immutable records."""
    data = _object(
        value, {"schema_version", "task_type", "class_map", "samples"}, "manifest"
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("manifest: supported schema_version is 1")
    task = data["task_type"]
    if task not in ("classification", "detection"):
        raise ValueError("manifest: task_type must be classification or detection")
    classes = data["class_map"]
    if not isinstance(classes, dict) or not classes:
        raise ValueError(
            "class_map: expected a nonempty object mapping names to indices"
        )
    for name, index in classes.items():
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError(
                "class_map: names must be nonempty strings without surrounding whitespace"
            )
        if type(index) is not int:
            raise ValueError("class_map: indices must be integers")
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError("class_map: indices must be unique and contiguous from zero")
    rows = data["samples"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("samples: expected a nonempty list")
    samples = []
    seen = set()
    for position, row in enumerate(rows):
        location = f"samples[{position}]"
        row = _object(row, {"image_key", "split", "labels"}, location)
        key = row["image_key"]
        if not isinstance(key, str) or not key or not key.isprintable():
            raise ValueError(
                f"{location}: image_key must be a nonempty printable string"
            )
        path = PurePosixPath(key)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in key
            or ":" in key
            or path.as_posix() != key
        ):
            raise ValueError(
                f"{location}: image_key must be a normalized relative object key"
            )
        if key in seen:
            raise ValueError(f"{location}: duplicate image_key {key}")
        seen.add(key)
        if row["split"] not in SPLITS:
            raise ValueError(f"{location}: split must be train, val, or test")
        labels = row["labels"]
        if not isinstance(labels, list) or (
            task == "classification" and len(labels) != 1
        ):
            raise ValueError(
                f"{location}: labels must be a list; classification requires exactly one"
            )
        parsed = tuple(
            _label(label, task, len(classes), f"{location}.labels[{index}]")
            for index, label in enumerate(labels)
        )
        samples.append(Sample(key, row["split"], parsed))
    class_names = tuple(sorted(classes, key=classes.__getitem__))
    return Manifest(1, task, class_names, tuple(samples))


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON number: {value}")


def load_manifest(path: str | Path) -> Manifest:
    """Load a UTF-8 JSON manifest without fetching images or importing ML libraries."""
    return parse_manifest_json(Path(path).read_text(encoding="utf-8"))


def parse_manifest_json(contents: str) -> Manifest:
    """Validate the same in-memory JSON text that a caller hashes for provenance."""
    return parse_manifest(
        json.loads(
            contents, object_pairs_hook=_unique_keys, parse_constant=_reject_constant
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Invalid manifest: {error}\n")
    print(
        json.dumps(
            {
                "task_type": manifest.task_type,
                "class_map": dict(manifest.class_map),
                "images_by_split": {
                    split: len(manifest.samples_for(split)) for split in SPLITS
                },
                "labels": sum(len(sample.labels) for sample in manifest.samples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
