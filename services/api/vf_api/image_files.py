"""Decode the same JPEG/PNG modes and orientation accepted by training."""

from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_PIXELS = 20_000_000
MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG"}


def decode_image(data: bytes, mime: str | None = None) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in MIME_FORMATS.values():
                    raise ValueError("Only JPEG and PNG images are supported")
                if mime is not None and image.format != MIME_FORMATS[mime]:
                    raise ValueError(
                        "Declared content type does not match image encoding"
                    )
                if image.mode not in ("RGB", "L") or getattr(
                    image, "is_animated", False
                ):
                    raise ValueError("Images must be single-frame RGB or grayscale")
                if (
                    min(image.size) < 10
                    or image.width * image.height > MAX_IMAGE_PIXELS
                ):
                    raise ValueError(
                        "Images must be at least 10x10 and at most 20 megapixels"
                    )
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("Normalize EXIF orientation before uploading")
                image.load()
                return image.copy()
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ValueError("Invalid, truncated or oversized image") from None


def make_thumbnail(data: bytes) -> bytes:
    with decode_image(data) as image:
        thumbnail = image.convert("RGB")
    try:
        thumbnail.thumbnail((320, 320), Image.Resampling.LANCZOS)
        output = BytesIO()
        thumbnail.save(output, format="JPEG", quality=85)
        return output.getvalue()
    finally:
        thumbnail.close()
