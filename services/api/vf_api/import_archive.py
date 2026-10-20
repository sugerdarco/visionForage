"""Read bounded ZIP members without extracting paths onto the filesystem."""

from pathlib import PurePosixPath
import stat
import struct
from typing import BinaryIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from vf_api.import_schemas import InvalidDataset, archive_path
from vf_api.storage import MAX_UPLOAD_BYTES

MAX_ENTRIES = 5000
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def check_directory(source: BinaryIO) -> None:
    # Bound entry count before ZipFile allocates an object for every central entry.
    source.seek(0, 2)
    size = source.tell()
    source.seek(max(0, size - 65557))
    tail = source.read()
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < 22:
        raise InvalidDataset("Invalid ZIP directory")
    _, disk, start_disk, local_count, count, length, start, comment = struct.unpack(
        "<4s4H2LH", tail[offset : offset + 22]
    )
    if (
        disk
        or start_disk
        or local_count != count
        or count > MAX_ENTRIES
        or len(tail) - offset != 22 + comment
        or start + length != size - len(tail) + offset
        or start == 0xFFFFFFFF
        or length == 0xFFFFFFFF
    ):
        raise InvalidDataset(
            "ZIP must be single-volume, non-ZIP64 and at most 5,000 entries"
        )
    source.seek(start)
    for _ in range(count):
        header = source.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            raise InvalidDataset("Invalid ZIP central directory")
        fields = struct.unpack("<4s6H3L5H2L", header)
        name_length, extra_length, comment_length = fields[10:13]
        if name_length > 2000 or fields[13] != 0:
            raise InvalidDataset("ZIP path or disk number is unsupported")
        source.seek(name_length + extra_length + comment_length, 1)
    if source.tell() != start + length:
        raise InvalidDataset("ZIP entry count does not match the directory")
    source.seek(0)


class Archive:
    def __init__(self, source: BinaryIO, root: str = ""):
        check_directory(source)
        self.zip = ZipFile(source)
        try:
            self.files = {}
            seen = set()
            total = 0
            entries = self.zip.infolist()
            if len(entries) > MAX_ENTRIES:
                raise InvalidDataset("ZIP contains too many entries")
            for entry in entries:
                if entry.orig_filename != entry.filename:
                    raise InvalidDataset("ZIP paths cannot contain null characters")
                name = archive_path(
                    entry.filename[:-1] if entry.is_dir() else entry.filename
                )
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise InvalidDataset(
                        "ZIP links and special files are not supported"
                    )
                if name in seen:
                    raise InvalidDataset("Duplicate ZIP paths are not supported")
                seen.add(name)
                if entry.flag_bits & 1 or entry.compress_type not in (
                    ZIP_STORED,
                    ZIP_DEFLATED,
                ):
                    raise InvalidDataset("Use an unencrypted stored or deflated ZIP")
                total += entry.file_size
                if total > MAX_EXPANDED_BYTES or entry.file_size > MAX_UPLOAD_BYTES:
                    raise InvalidDataset(
                        "ZIP exceeds expanded or individual file limits"
                    )
                if entry.file_size > max(1, entry.compress_size) * 200:
                    raise InvalidDataset("ZIP member compression ratio exceeds 200:1")
                if entry.is_dir():
                    continue
                if kind == stat.S_IFDIR:
                    raise InvalidDataset("Invalid ZIP directory entry")
                if root:
                    if not name.startswith(root + "/"):
                        continue
                    name = name[len(root) + 1 :]
                self.files[name] = entry
            for name in self.files:
                if any(
                    str(parent) in self.files for parent in PurePosixPath(name).parents
                ):
                    raise InvalidDataset("ZIP file/directory paths conflict")
            self.images = {
                name: IMAGE_TYPES[PurePosixPath(name).suffix.lower()]
                for name in self.files
                if PurePosixPath(name).suffix.lower() in IMAGE_TYPES
            }
            if not 1 <= len(self.images) <= 1000:
                raise InvalidDataset(
                    "Import must contain between 1 and 1,000 JPEG/PNG images"
                )
        except Exception:
            self.zip.close()
            raise

    def close(self) -> None:
        self.zip.close()

    def read(self, name: str, limit: int = MAX_METADATA_BYTES) -> bytes:
        if name not in self.files:
            raise InvalidDataset(f"Missing archive member: {name}")
        entry = self.files[name]
        if entry.file_size > limit:
            raise InvalidDataset(f"Archive member exceeds its size limit: {name}")
        with self.zip.open(entry) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit or len(data) != entry.file_size:
            raise InvalidDataset(f"Invalid archive member size: {name}")
        return data

    def check_used(self, used: set[str]) -> None:
        allowed = {"README.md", "README.txt", "LICENSE", "LICENSE.txt", ".DS_Store"}
        extra = {
            name
            for name in self.files.keys() - used
            if PurePosixPath(name).name not in allowed
        }
        if extra:
            raise InvalidDataset(
                f"Unrecognized or unreferenced archive member: {min(extra)}"
            )
