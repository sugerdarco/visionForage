"""Tiny in-memory dataset archives for parser and storage integration tests."""

from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image

from vf_api.import_archive import Archive
from vf_api.import_formats import parse_dataset
from vf_api.import_schemas import ImportClaim


def picture(color=0):
    output = BytesIO()
    with Image.new("RGB", (100, 50), color=color) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def zipped(files):
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def coco_document():
    return {
        "categories": [{"id": 7, "name": "scratch"}],
        "images": [{"id": 12, "file_name": "a.png", "width": 100, "height": 50}],
        "annotations": [
            {
                "id": 20,
                "image_id": 12,
                "category_id": 7,
                "bbox": [10, 5, 40, 20],
                "iscrowd": 0,
            }
        ],
    }


def dataset_files(format):
    if format == "imagefolder":
        return {"scratch/a.png": picture()}
    if format == "yolo":
        return {
            "data.yaml": "names: [scratch]\n",
            "images/a.png": picture(),
            "labels/a.txt": "0 0.3 0.3 0.4 0.4\n",
        }
    return {"annotations.json": json.dumps(coco_document()), "images/a.png": picture()}


def claim_for(data, format, **kwargs):
    return ImportClaim(
        format=format, size=len(data), sha256=hashlib.sha256(data).hexdigest(), **kwargs
    )


@contextmanager
def opened(data, format="imagefolder", **kwargs):
    claim = claim_for(data, format, **kwargs)
    with BytesIO(data) as source:
        archive = Archive(source, claim.root)
        try:
            yield parse_dataset(archive, claim)
        finally:
            archive.close()
