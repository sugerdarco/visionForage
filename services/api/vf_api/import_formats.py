"""Convert common dataset formats into verified images and named labels."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import PurePosixPath

from pydantic import TypeAdapter
import yaml

from vf_api.annotation_schemas import AnnotationInput, BoundingBox
from vf_api.image_files import decode_image
from vf_api.import_archive import Archive
from vf_api.import_schemas import ImportClaim, InvalidDataset, archive_path
from vf_api.project_schemas import Name
from vf_api.storage import MAX_UPLOAD_BYTES

MAX_LABELS = 50_000
MAX_TOTAL_PIXELS = 500_000_000


@dataclass(frozen=True, order=True)
class Label:
    name: str
    bbox: tuple[float, ...] = ()


@dataclass
class Sample:
    path: str
    labels: list[Label] = field(default_factory=list)
    width: int = 0
    height: int = 0
    sha256: str = ""
    mime: str = ""


@dataclass
class Dataset:
    names: list[str]
    samples: list[Sample]


def class_names(values: list) -> list[str]:
    if not 1 <= len(values) <= 100:
        raise InvalidDataset("Import must define between 1 and 100 classes")
    names = [TypeAdapter(Name).validate_python(value) for value in values]
    if len(set(names)) != len(names):
        raise InvalidDataset("Class names must be unique after trimming whitespace")
    return names


def box(values) -> tuple[float, ...]:
    return AnnotationInput.within_image(
        TypeAdapter(BoundingBox).validate_python(values)
    )


def unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, (str, int)) or isinstance(key, bool) or key in result:
            raise InvalidDataset("Metadata contains an invalid or duplicate key")
        result[key] = value
    return result


class MetadataLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        return unique_mapping(
            [
                (
                    self.construct_object(key, deep=deep),
                    self.construct_object(value, deep=deep),
                )
                for key, value in node.value
            ]
        )


def imagefolder(archive: Archive) -> Dataset:
    paths = sorted(archive.images)
    split = all(path.split("/")[0] in {"train", "val", "test"} for path in paths)
    raw_names = []
    for path in paths:
        parts = path.split("/")
        if len(parts) < (3 if split else 2):
            raise InvalidDataset("ImageFolder images must be inside class directories")
        raw_names.append(parts[1 if split else 0])
    originals = sorted(set(raw_names))
    names = class_names(originals)
    lookup = dict(zip(originals, names))
    archive.check_used(set(paths))
    return Dataset(
        names,
        [Sample(path, [Label(lookup[name])]) for path, name in zip(paths, raw_names)],
    )


def yolo(archive: Archive) -> Dataset:
    content = archive.read("data.yaml").decode("utf-8-sig")
    for index, token in enumerate(yaml.scan(content)):
        if index > 20_000 or isinstance(token, (yaml.AliasToken, yaml.AnchorToken)):
            raise InvalidDataset(
                "YOLO metadata cannot contain aliases or excessive tokens"
            )
    metadata = yaml.load(content, Loader=MetadataLoader)
    if not isinstance(metadata, dict):
        raise InvalidDataset("data.yaml must contain a class names mapping")
    values = metadata.get("names")
    if isinstance(values, dict):
        if any(type(key) is not int for key in values) or set(values) != set(
            range(len(values))
        ):
            raise InvalidDataset(
                "YOLO class indices must be contiguous integers starting at zero"
            )
        values = [values[index] for index in range(len(values))]
    if not isinstance(values, list):
        raise InvalidDataset("YOLO names must be a list or integer mapping")
    names = class_names(values)
    if "nc" in metadata and (
        type(metadata["nc"]) is not int or metadata["nc"] != len(names)
    ):
        raise InvalidDataset("YOLO nc must match the number of class names")
    used = {"data.yaml"}
    samples = []
    label_paths = set()
    total_labels = 0
    for path in sorted(archive.images):
        if not path.startswith("images/"):
            raise InvalidDataset("YOLO images must be under images/")
        label_path = str(
            PurePosixPath("labels/" + path[len("images/") :]).with_suffix(".txt")
        )
        if label_path in label_paths:
            raise InvalidDataset("YOLO images cannot share a label file stem")
        label_paths.add(label_path)
        sample = Sample(path)
        used.add(path)
        if label_path in archive.files:
            used.add(label_path)
            for line in archive.read(label_path).decode("utf-8-sig").splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                if (
                    len(fields) != 5
                    or not fields[0].isascii()
                    or not fields[0].isdigit()
                ):
                    raise InvalidDataset(
                        "YOLO rows require class_index center_x center_y width height"
                    )
                class_id = int(fields[0])
                if class_id >= len(names):
                    raise InvalidDataset("Unknown YOLO class index")
                cx, cy, width, height = map(float, fields[1:])
                sample.labels.append(
                    Label(
                        names[class_id],
                        box([cx - width / 2, cy - height / 2, width, height]),
                    )
                )
                if len(sample.labels) > 1000:
                    raise InvalidDataset("An image cannot exceed 1,000 boxes")
                total_labels += 1
                if total_labels > MAX_LABELS:
                    raise InvalidDataset("Import cannot exceed 50,000 labels")
        samples.append(sample)
    archive.check_used(used)
    return Dataset(names, samples)


def integer(value, description: str) -> int:
    if type(value) is not int or value < 0:
        raise InvalidDataset(f"{description} must be a nonnegative integer")
    return value


def records(document: dict, key: str, limit: int) -> list[dict]:
    rows = document.get(key)
    if (
        not isinstance(rows, list)
        or len(rows) > limit
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise InvalidDataset(f"Invalid or oversized COCO {key} list")
    ids = [integer(row.get("id"), f"COCO {key} ID") for row in rows]
    if len(set(ids)) != len(ids):
        raise InvalidDataset(f"Duplicate COCO {key} ID")
    return rows


def coco(archive: Archive, annotations_file: str) -> Dataset:
    def reject_constant(value):
        raise InvalidDataset("COCO metadata requires finite numbers")

    document = json.loads(
        archive.read(annotations_file).decode("utf-8-sig"),
        object_pairs_hook=unique_mapping,
        parse_constant=reject_constant,
    )
    if not isinstance(document, dict):
        raise InvalidDataset("COCO annotations must be a JSON object")
    categories = sorted(records(document, "categories", 100), key=lambda row: row["id"])
    names = class_names([row.get("name") for row in categories])
    category_names = {row["id"]: name for row, name in zip(categories, names)}
    samples = {}
    used = {annotations_file}
    for row in records(document, "images", 1000):
        if not isinstance(row.get("file_name"), str):
            raise InvalidDataset("COCO file_name must be a relative image path")
        name = archive_path(row["file_name"])
        matches = [
            candidate
            for candidate in (name, "images/" + name)
            if candidate in archive.images
        ]
        if len(matches) != 1 or matches[0] in used:
            raise InvalidDataset("COCO image path is missing, ambiguous or duplicated")
        path = matches[0]
        used.add(path)
        samples[row["id"]] = Sample(
            path,
            width=integer(row.get("width"), "COCO width"),
            height=integer(row.get("height"), "COCO height"),
        )
        if samples[row["id"]].width == 0 or samples[row["id"]].height == 0:
            raise InvalidDataset("COCO dimensions must be positive")
    for row in records(document, "annotations", MAX_LABELS):
        image_id = integer(row.get("image_id"), "COCO image reference")
        category_id = integer(row.get("category_id"), "COCO category reference")
        if image_id not in samples or category_id not in category_names:
            raise InvalidDataset("Unknown COCO image or category reference")
        if row.get("iscrowd", 0) != 0 or row.get("ignore", 0) != 0:
            raise InvalidDataset("Crowd and ignored COCO annotations are not supported")
        sample = samples[image_id]
        raw = row.get("bbox")
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or any(type(value) not in (int, float) for value in raw)
        ):
            raise InvalidDataset("COCO bbox requires four numeric pixel coordinates")
        x, y, width, height = raw
        sample.labels.append(
            Label(
                category_names[category_id],
                box(
                    [
                        x / sample.width,
                        y / sample.height,
                        width / sample.width,
                        height / sample.height,
                    ]
                ),
            )
        )
        if len(sample.labels) > 1000:
            raise InvalidDataset("An image cannot exceed 1,000 boxes")
    archive.check_used(used)
    return Dataset(names, sorted(samples.values(), key=lambda row: row.path))


def parse_dataset(archive: Archive, claim: ImportClaim) -> Dataset:
    if claim.format == "imagefolder":
        dataset = imagefolder(archive)
    elif claim.format == "yolo":
        dataset = yolo(archive)
    else:
        dataset = coco(archive, claim.annotations_file)
    if (
        not dataset.samples
        or sum(len(sample.labels) for sample in dataset.samples) > MAX_LABELS
    ):
        raise InvalidDataset("Import must contain images and at most 50,000 labels")
    pixels = 0
    hashes = {}
    for sample in dataset.samples:
        data = archive.read(sample.path, MAX_UPLOAD_BYTES)
        sample.mime = archive.images[sample.path]
        with decode_image(data, sample.mime) as image:
            if claim.format == "coco" and image.size != (sample.width, sample.height):
                raise InvalidDataset(
                    "COCO dimensions do not match decoded image dimensions"
                )
            sample.width, sample.height = image.size
        pixels += sample.width * sample.height
        if pixels > MAX_TOTAL_PIXELS:
            raise InvalidDataset("Import exceeds 500 million decoded pixels")
        sample.sha256 = hashlib.sha256(data).hexdigest()
        sample.labels.sort()
        if sample.sha256 in hashes and hashes[sample.sha256] != sample.labels:
            raise InvalidDataset(
                "Identical image bytes have conflicting labels in the archive"
            )
        hashes[sample.sha256] = sample.labels
    return dataset
