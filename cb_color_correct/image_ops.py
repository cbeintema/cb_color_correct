from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .lut import CubeLut, apply_lut


ArrayF = np.ndarray


def _to_float01(rgb8: np.ndarray) -> ArrayF:
    if rgb8.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {rgb8.dtype}")
    return rgb8.astype(np.float32) / 255.0


def _to_uint8(rgb01: ArrayF) -> np.ndarray:
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return (rgb01 * 255.0 + 0.5).astype(np.uint8)


def luminance(rgb01: ArrayF) -> ArrayF:
    # Rec.709 luma
    return rgb01[..., 0] * 0.2126 + rgb01[..., 1] * 0.7152 + rgb01[..., 2] * 0.0722


def blend(original01: ArrayF, filtered01: ArrayF, strength: float) -> ArrayF:
    strength = float(np.clip(strength, 0.0, 1.0))
    return original01 * (1.0 - strength) + filtered01 * strength


def apply_brightness(rgb01: ArrayF, brightness: float) -> ArrayF:
    # brightness in [-1, 1] as additive offset
    return rgb01 + float(brightness)


def apply_contrast(rgb01: ArrayF, contrast: float) -> ArrayF:
    # contrast in [-1, 1]; 0=no change
    c = float(contrast)
    if c == 0:
        return rgb01
    factor = 1.0 + c
    return (rgb01 - 0.5) * factor + 0.5


def apply_exposure(rgb01: ArrayF, exposure: float) -> ArrayF:
    # exposure in stops
    return rgb01 * (2.0 ** float(exposure))


def apply_channel_multipliers(rgb01: ArrayF, r: float = 1.0, g: float = 1.0, b: float = 1.0) -> ArrayF:
    mul = np.array([r, g, b], dtype=np.float32)
    return rgb01 * mul


def apply_gamma(rgb01: ArrayF, gamma: float) -> ArrayF:
    g = float(gamma)
    if g <= 0:
        return rgb01
    return np.power(np.clip(rgb01, 0.0, 1.0), 1.0 / g)


def apply_levels(rgb01: ArrayF, black: float = 0.0, white: float = 1.0, gamma: float = 1.0) -> ArrayF:
    # black/white in [0,1]
    b = float(black)
    w = float(white)
    if w <= b + 1e-6:
        return rgb01
    out = (rgb01 - b) / (w - b)
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = apply_gamma(out, gamma)
    return out


def _lut_from_points(points: Sequence[tuple[float, float]]) -> np.ndarray:
    # points in 0..1 space; monotonic x.
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    grid = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.interp(grid, xs, ys).astype(np.float32)
    return np.clip(lut, 0.0, 1.0)


def apply_curve(rgb01: ArrayF, points: Sequence[tuple[float, float]]) -> ArrayF:
    lut = _lut_from_points(points)
    idx = np.clip((rgb01 * 255.0 + 0.5).astype(np.int32), 0, 255)
    return lut[idx]


def apply_saturation(rgb01: ArrayF, saturation: float) -> ArrayF:
    # saturation in [-1, 1]
    s = float(saturation)
    if s == 0:
        return rgb01
    l = luminance(rgb01)[..., None]
    factor = 1.0 + s
    return l + (rgb01 - l) * factor


def apply_vibrance(rgb01: ArrayF, vibrance: float) -> ArrayF:
    # Boosts low-sat areas more than high-sat. vibrance in [-1, 1]
    v = float(vibrance)
    if v == 0:
        return rgb01
    l = luminance(rgb01)[..., None]
    delta = rgb01 - l
    # Rough saturation proxy: max channel distance from luma
    sat = np.max(np.abs(delta), axis=-1, keepdims=True)
    boost = 1.0 + v * (1.0 - np.clip(sat * 3.0, 0.0, 1.0))
    return l + delta * boost


def apply_hue_shift(rgb01: ArrayF, degrees: float) -> ArrayF:
    # Simple hue rotation in YIQ-like space for speed.
    # This is an approximation but works well for creative filters.
    theta = np.deg2rad(float(degrees))
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)

    # RGB -> YIQ
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b

    i2 = i * cos_t - q * sin_t
    q2 = i * sin_t + q * cos_t

    # YIQ -> RGB
    r2 = y + 0.956 * i2 + 0.621 * q2
    g2 = y - 0.272 * i2 - 0.647 * q2
    b2 = y - 1.106 * i2 + 1.703 * q2
    out = np.stack([r2, g2, b2], axis=-1).astype(np.float32)
    return out


def apply_split_tone(
    rgb01: ArrayF,
    shadows_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlights_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    balance: float = 0.0,
    amount: float = 0.3,
) -> ArrayF:
    # Adds color to shadows/highlights based on luminance.
    amt = float(amount)
    if amt == 0:
        return rgb01

    l = luminance(rgb01)
    # balance shifts midpoint: negative favors shadows, positive favors highlights
    mid = 0.5 + float(balance) * 0.25
    # Smooth masks
    shadows_mask = np.clip((mid - l) * 2.0, 0.0, 1.0)
    highlights_mask = np.clip((l - mid) * 2.0, 0.0, 1.0)

    sh = np.array(shadows_rgb, dtype=np.float32)[None, None, :]
    hi = np.array(highlights_rgb, dtype=np.float32)[None, None, :]

    out = rgb01 + (sh * shadows_mask[..., None] + hi * highlights_mask[..., None]) * amt
    return out


@dataclass(frozen=True)
class FilterParams:
    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    hue_degrees: float = 0.0
    levels_black: float = 0.0
    levels_white: float = 1.0
    levels_gamma: float = 1.0
    curve_points: tuple[tuple[float, float], ...] | None = None
    channel_mul: tuple[float, float, float] = (1.0, 1.0, 1.0)
    split_shadows: tuple[float, float, float] = (0.0, 0.0, 0.0)
    split_highlights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    split_balance: float = 0.0
    split_amount: float = 0.0
    lut: CubeLut | None = None


def apply_params(rgb01: ArrayF, params: FilterParams) -> ArrayF:
    out = rgb01
    if params.exposure:
        out = apply_exposure(out, params.exposure)
    if params.brightness:
        out = apply_brightness(out, params.brightness)
    if params.contrast:
        out = apply_contrast(out, params.contrast)
    if params.hue_degrees:
        out = apply_hue_shift(out, params.hue_degrees)
    if params.saturation:
        out = apply_saturation(out, params.saturation)
    if params.vibrance:
        out = apply_vibrance(out, params.vibrance)
    if params.channel_mul != (1.0, 1.0, 1.0):
        out = apply_channel_multipliers(out, *params.channel_mul)
    if params.levels_black != 0.0 or params.levels_white != 1.0 or params.levels_gamma != 1.0:
        out = apply_levels(out, params.levels_black, params.levels_white, params.levels_gamma)
    if params.curve_points is not None:
        out = apply_curve(out, params.curve_points)
    if params.split_amount:
        out = apply_split_tone(
            out,
            shadows_rgb=params.split_shadows,
            highlights_rgb=params.split_highlights,
            balance=params.split_balance,
            amount=params.split_amount,
        )
    if params.lut is not None:
        out = apply_lut(out, params.lut)
    return np.clip(out, 0.0, 1.0)


def process_rgb8(rgb8: np.ndarray, params: FilterParams, strength: float) -> np.ndarray:
    rgb01 = _to_float01(rgb8)
    filtered01 = apply_params(rgb01, params)
    mixed01 = blend(rgb01, filtered01, strength)
    return _to_uint8(mixed01)


def resize_for_preview(rgb8: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = rgb8.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return rgb8
    scale = max_side / float(side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    # Simple area-like downsample using slicing+reshape if divisible would be faster;
    # for general case we use Pillow in loader.
    # This is a fallback; normally we resize via PIL before converting to numpy.
    ys = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return rgb8[ys[:, None], xs[None, :]]
