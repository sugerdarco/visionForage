# Import labeled datasets

ImageFolder, YOLO detection and COCO detection ZIPs can now populate a project.
The flow is: browser uploads the archive directly to private MinIO storage;
FastAPI downloads it into a temporary file, validates its contents, and commits
classes, images and annotations to PostgreSQL. The existing Redis/Celery CPU
worker generates thumbnails after that commit. No GPU or training library is
used by the importer.

This is a bounded, synchronous import. It keeps error reporting and retries
simple while the application is small. Imports larger than the limits below
will need a background-job workflow. [Dataset-version creation](DATASET_VERSIONS.md)
now freezes these images and labels into the manifest that training will consume.

## Upload and complete

Use an access bearer token and a project owned by that account. All paths below
are under `/api/v1`. First compute the ZIP's byte size and SHA-256, then request:

```http
POST /projects/{project_id}/imports/presign
Content-Type: application/json

{
  "format": "imagefolder",
  "size": 12345,
  "sha256": "<64 lowercase hexadecimal characters>",
  "root": ""
}
```

Replace the example size and digest with the actual file values. `format` is
`imagefolder`, `yolo` or `coco`. ImageFolder requires a classification project;
YOLO and COCO require detection. `root` optionally names a directory inside the
ZIP, for example `my-dataset`; it defaults to the archive root. No automatic
wrapper-folder guessing occurs. COCO also accepts `annotations_file`, a path
relative to that root, defaulting to `annotations.json`.
Both option paths are limited to 200 UTF-8 bytes so signed grants remain bounded.

The response contains `upload_token`, `upload_url`, `headers` and
`expires_in: 900`. PUT the raw ZIP bytes to `upload_url` using the returned
headers, including `Content-Type: application/zip` and the SHA-256 checksum.
Do not send multipart form data or the API bearer token to MinIO. The byte length
must match the signed size. Then call:

```http
POST /projects/{project_id}/imports
Content-Type: application/json

{"upload_token": "<token returned by presign>"}
```

Completion may take longer than a normal API request because all images are
decoded. It returns 200 with:

```json
{
  "project_id": "<project UUID>",
  "created_images": 2,
  "labeled_images": 1,
  "unchanged_images": 3,
  "duplicate_files": 1,
  "created_classes": 1,
  "items": [
    {"path": "ok/sample.png", "image_id": "<image UUID>", "action": "created"}
  ]
}
```

This abbreviated example shows one item; the real response includes one per
source image. The four image/file counts sum to the number of items.
`labeled` means an existing unlabeled image received imported labels. `duplicate`
means another file in this archive has the same image bytes and labels. Every
imported image becomes `labeled`, including detection backgrounds with no boxes.

Foreign/missing project IDs return 404; invalid/expired grants return 400;
missing staging objects or conflicting existing labels return 409. Format,
image and limit violations return 422. Storage outages return 503. One import
per API process runs at a time; another receives 503 with `Retry-After: 5`.

## Supported layouts

ImageFolder accepts class directories with JPEG/PNG files, including nested
subdirectories:

```text
ok/part-001.png
defect/batch-a/part-002.jpg
```

It also accepts `train/{class}/...`, `val/{class}/...` and `test/{class}/...` when
all images use those top-level split directories. Class names come from directory
names, with surrounding whitespace trimmed. Duplicate normalized class names
are rejected. New classes are appended in sorted name order.

YOLO accepts this layout, optionally with matching subdirectories under images
and labels:

```text
data.yaml
images/train/part-001.png
labels/train/part-001.txt
```

```yaml
names:
  0: scratch
  1: dent
nc: 2
```

`names` can also be a list. Dictionary indices must be contiguous integers from
zero; optional `nc` must match the number of names. Each nonempty label line is
`class_index center_x center_y width height`, with normalized coordinates. The
importer converts centers to the API's normalized top-left coordinates.
An empty **or missing** label file represents a background image, following the
[Ultralytics detection format](https://docs.ultralytics.com/datasets/detect/).
Check missing files before importing: they will become labeled negatives.

Only axis-aligned detection boxes are supported. Segmentation, pose and oriented
box rows are rejected. Orphan label files and image names sharing a label-file
stem, such as `a.jpg` and `a.png`, are rejected. YAML is safely parsed with no
aliases, anchors or executable tags. The importer uses `names` and optional
`nc`; it does not follow YAML `path`, split paths, image-list files, download
scripts or remote URLs. Put all desired images under `images/`.

COCO accepts a JSON document containing `images`, `categories` and `annotations`:

```text
annotations.json
images/part-001.png
```

```json
{
  "images": [{"id": 12, "file_name": "part-001.png", "width": 100, "height": 50}],
  "categories": [{"id": 7, "name": "scratch"}],
  "annotations": [{"id": 20, "image_id": 12, "category_id": 7,
                   "bbox": [10, 5, 40, 20], "iscrowd": 0}]
}
```

COCO boxes use pixel `[x, y, width, height]`; the example becomes
`[0.1, 0.1, 0.4, 0.4]`. The importer verifies declared image dimensions against
decoded pixels. Sparse category IDs are mapped by category name, not copied into
the project's model indices. This follows the box representation in the
[COCO reference implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).

Each `file_name` resolves either directly under the selected root or under its
`images/` directory; ambiguous matches are rejected. A custom JSON path can be
supplied, such as `annotations/instances_train.json`. All selected images must
be referenced by that document. IDs must be unique nonnegative integers within
each list, and annotation references must exist. Images with no annotations
are labeled negatives. Crowd and ignored annotations are rejected because the
current training contract cannot represent their evaluation rules. Segmentation,
area and other metadata are not imported; a valid detection box is required.

All formats pool the selected images into the live project. **Source train/val/test
assignments are not retained.** [Dataset-version creation](DATASET_VERSIONS.md) chooses
and freezes splits. This endpoint does not recreate a benchmark's official split.

## Duplicates, labels and class indices

Classes match by exact, trimmed, case-sensitive name. Existing UUIDs, colors and
indices remain unchanged. New names append to the project's class map using the
same project lock as manual class creation; new classes use `#3b82f6` initially.
Source YOLO indices and COCO category IDs are translated through those names.

Image identity is the SHA-256 of original file bytes within one project, matching
ordinary uploads. Identical bytes and labels inside a ZIP share one image row;
conflicting labels for identical bytes reject the entire archive. Repeated boxes
within a valid detection label set are preserved.

For an existing image, the importer can fill an empty `unlabeled` record. An
already labeled image with identical human labels is left unchanged, preserving
annotation IDs. Different labels, model suggestions or `in_review` status cause
a 409 for the entire import. Use the annotation editor to resolve those conflicts;
an archive never silently overwrites reviewed work. Matching uses exact stored
box values, so differently rounded versions of a box can conflict.

The project lock serializes class/index allocation and image registration.
Existing matching image rows are also locked, so a concurrent annotation edit
is rechecked after that edit commits. All conflicts are checked before writing
new originals. Labels imported from files have `source: human`, matching the
current annotation model; file-level annotation provenance is not retained.

## Limits and recovery

| Limit | Value |
| --- | --- |
| ZIP upload | 64 MiB |
| Expanded ZIP total | 256 MiB |
| ZIP entries, including directories | 5,000 |
| Individual member | 20 MiB |
| Individual metadata/label file | 4 MiB |
| Compression ratio per member | 200:1 |
| Images | 1–1,000 |
| Class names in one import | 1–100 |
| Boxes per image / labels per import | 1,000 / 50,000 |
| Total decoded pixels | 500 million |

Images retain the [ordinary upload rules](UPLOADS.md): JPEG/PNG, RGB or grayscale,
single frame, normalized EXIF orientation, at least 10 × 10 pixels and at most
20 megapixels each. Bounding boxes follow [annotation validation](ANNOTATIONS.md).
ZIPs must be unencrypted, single-volume, non-ZIP64, stored or deflated. Traversal,
absolute paths, duplicate paths, links and special files are rejected. Members
are read directly, never extracted using their archive paths. Common README,
LICENSE and `.DS_Store` files are ignored; other unrecognized files are rejected.
Members outside an explicitly selected `root` are not imported, but still count
toward archive limits and undergo path/type checks.

The archive grant binds its owner, project, digest, format and paths, and uses a
separate JWT audience from image uploads and login. Completion verifies the full
size and SHA-256 before parsing. The temporary local file is closed on success
and failure. The staging object stays in `uploads/` for retry and is covered by
the existing one-day lifecycle rule; configure it with `make storage-config`.
Within the 15-minute grant window, repeat completion after a lost response.
After expiry, obtain a new grant and upload again. Repeating an unchanged import
creates no additional rows, though its response counts change to `unchanged`.

Database failures roll back the entire batch. MinIO and PostgreSQL have no shared
transaction: an interrupted object write/registration sequence can leave an
unreferenced original. Deterministic keys let a retry reuse it, but general orphan
cleanup is still pending. Queue outages leave thumbnails pending; the existing
periodic recovery task picks them up. No schema migration is needed.

## Verification

`make test-db` includes archive parser checks. `make test-uploads` includes real
MinIO signed uploads, PostgreSQL transactions and a Redis/Celery CPU consumer.
New coverage includes all three formats, negative images, conversion, invalid
archives/metadata, limits, ownership, stable class indices, duplicate/retry behavior,
concurrent label edits, queue outages and rollback after an object-store failure.

Validated for this milestone: 51 API/database/parser tests, 21 storage/worker
integration tests and 110 ML/setup regression tests passed, together with lint.
