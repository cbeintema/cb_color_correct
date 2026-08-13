from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class CensorCircle:
    center_x: float
    center_y: float
    radius: float


def apply_censor_blur(
    rgb8: np.ndarray,
    circles: Iterable[CensorCircle],
    blur_radius: float,
) -> np.ndarray:
    if rgb8.ndim != 3 or rgb8.shape[2] != 3 or rgb8.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB array")

    blur = float(blur_radius)
    if not isfinite(blur):
        raise ValueError("Blur radius must be finite")

    circle_values = tuple(circles)
    if not circle_values or blur <= 0.0:
        return rgb8.copy()

    image = Image.fromarray(rgb8, mode="RGB")
    width, height = image.size

    for circle in circle_values:
        center_x = float(circle.center_x) * width
        center_y = float(circle.center_y) * height
        radius = float(circle.radius) * width
        if not all(isfinite(value) for value in (center_x, center_y, radius)) or radius <= 0.0:
            continue

        padding = 3.0 * blur
        left = max(0, int(floor(center_x - radius - padding)))
        top = max(0, int(floor(center_y - radius - padding)))
        right = min(width, int(ceil(center_x + radius + padding)))
        bottom = min(height, int(ceil(center_y + radius + padding)))
        if right <= left or bottom <= top:
            continue

        crop = image.crop((left, top, right, bottom))
        blurred = crop.filter(ImageFilter.GaussianBlur(blur))
        mask = Image.new("L", crop.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse(
            (
                center_x - radius - left,
                center_y - radius - top,
                center_x + radius - left,
                center_y + radius - top,
            ),
            fill=255,
        )
        composited = Image.composite(blurred, crop, mask)
        image.paste(composited, (left, top))

    return np.asarray(image, dtype=np.uint8).copy()
