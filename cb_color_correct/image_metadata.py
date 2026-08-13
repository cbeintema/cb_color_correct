from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


_FORMAT_BY_SUFFIX = {
    ".bmp": "BMP",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}


def _metadata_free_copy(image: Image.Image) -> Image.Image:
    clean = image.copy()
    clean.info.clear()
    clean.getexif().clear()
    return clean


def save_metadata_free_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = _metadata_free_copy(image)
    try:
        image_format = _FORMAT_BY_SUFFIX.get(output_path.suffix.lower())
        clean.save(output_path, format=image_format)
    finally:
        clean.close()


def save_metadata_free_rgb8(rgb8: np.ndarray, output_path: Path) -> None:
    if rgb8.ndim != 3 or rgb8.shape[2] != 3 or rgb8.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB array")
    image = Image.fromarray(rgb8, mode="RGB")
    try:
        save_metadata_free_image(image, output_path)
    finally:
        image.close()


def save_metadata_free_bytes(image_bytes: bytes, output_path: Path) -> None:
    with Image.open(BytesIO(image_bytes)) as image:
        mode = "RGBA" if "A" in image.getbands() else "RGB"
        clean = image.convert(mode)
        try:
            save_metadata_free_image(clean, output_path)
        finally:
            clean.close()
