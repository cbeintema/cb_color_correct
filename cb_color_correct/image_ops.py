from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .lut import CubeLut, apply_lut

try:
    import pilgram2 as pilgram
except Exception:
    pilgram = None


ArrayF = np.ndarray


def _to_float01(rgb8: np.ndarray) -> ArrayF:
    if rgb8.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {rgb8.dtype}")
    return rgb8.astype(np.float32) / 255.0


def _to_uint8(rgb01: ArrayF) -> np.ndarray:
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return (rgb01 * 255.0 + 0.5).astype(np.uint8)


def _apply_pilgram_filter(rgb01: ArrayF, filter_name: str) -> ArrayF:
    if pilgram is None:
        return rgb01

    fn = getattr(pilgram, filter_name, None)
    if fn is None or not callable(fn):
        return rgb01

    rgb8 = _to_uint8(rgb01)
    pil_img = Image.fromarray(rgb8, mode="RGB")
    try:
        out_pil = fn(pil_img)
    except Exception:
        return rgb01
    out_rgb8 = np.asarray(out_pil.convert("RGB"), dtype=np.uint8)
    return _to_float01(out_rgb8)


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


def apply_curve_rgb(
    rgb01: ArrayF,
    points_r: Sequence[tuple[float, float]] | None,
    points_g: Sequence[tuple[float, float]] | None,
    points_b: Sequence[tuple[float, float]] | None,
) -> ArrayF:
    if points_r is None and points_g is None and points_b is None:
        return rgb01

    idx = np.clip((rgb01 * 255.0 + 0.5).astype(np.int32), 0, 255)
    out = rgb01.copy()

    if points_r is not None:
        lut_r = _lut_from_points(points_r)
        out[..., 0] = lut_r[idx[..., 0]]
    if points_g is not None:
        lut_g = _lut_from_points(points_g)
        out[..., 1] = lut_g[idx[..., 1]]
    if points_b is not None:
        lut_b = _lut_from_points(points_b)
        out[..., 2] = lut_b[idx[..., 2]]

    return out


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


def apply_temperature_tint(rgb01: ArrayF, temperature: float, tint: float) -> ArrayF:
    # Both inputs in [-1, 1].
    # Temperature: warm (+) shifts toward red, cool (-) toward blue.
    # Tint: magenta (+) vs green (-).
    t = float(temperature)
    ti = float(tint)
    if t == 0.0 and ti == 0.0:
        return rgb01

    # Keep it subtle and pleasant: multipliers roughly in ~[0.85, 1.15] at extremes.
    r = 1.0 + 0.18 * t + 0.08 * ti
    g = 1.0 - 0.12 * ti
    b = 1.0 - 0.18 * t + 0.08 * ti
    return apply_channel_multipliers(rgb01, r=r, g=g, b=b)


def apply_shadows_highlights(
    rgb01: ArrayF,
    shadows: float,
    highlights: float,
    balance: float = 0.0,
) -> ArrayF:
    # shadows/highlights in [-1, 1], balance in [-1, 1]
    sh = float(shadows)
    hi = float(highlights)
    if sh == 0.0 and hi == 0.0:
        return rgb01

    l = luminance(rgb01)
    mid = 0.5 + float(balance) * 0.15
    # Masks
    shadows_mask = np.clip((mid - l) * 2.2, 0.0, 1.0)
    highlights_mask = np.clip((l - mid) * 2.2, 0.0, 1.0)

    # Amount scaling: keep within a reasonable range
    sh_amt = sh * 0.35
    hi_amt = hi * 0.35
    out = rgb01 + shadows_mask[..., None] * sh_amt + highlights_mask[..., None] * hi_amt
    return out


def apply_blacks_whites(rgb01: ArrayF, blacks: float, whites: float) -> ArrayF:
    # blacks/whites in [-1, 1]
    b = float(blacks)
    w = float(whites)
    if b == 0.0 and w == 0.0:
        return rgb01

    l = luminance(rgb01)
    blacks_mask = np.clip((0.35 - l) * 3.0, 0.0, 1.0)
    whites_mask = np.clip((l - 0.65) * 3.0, 0.0, 1.0)

    b_amt = b * 0.30
    w_amt = w * 0.30
    out = rgb01 + blacks_mask[..., None] * b_amt + whites_mask[..., None] * w_amt
    return out


def _smoothstep(edge0: float, edge1: float, x: ArrayF) -> ArrayF:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_midtone_contrast(rgb01: ArrayF, amount: float) -> ArrayF:
    a = float(amount)
    if a == 0.0:
        return rgb01
    l = luminance(rgb01)
    # Midtone mask peaks around 0.5 and falls off toward shadows/highlights.
    mid = 1.0 - np.clip(np.abs(l - 0.5) * 2.2, 0.0, 1.0)
    factor = 1.0 + a * 0.9 * mid
    return (rgb01 - 0.5) * factor[..., None] + 0.5


def apply_dehaze(rgb01: ArrayF, amount: float) -> ArrayF:
    a = float(amount)
    if a == 0.0:
        return rgb01
    out = rgb01
    out = apply_midtone_contrast(out, a * 0.65)
    out = apply_contrast(out, a * 0.20)
    out = apply_blacks_whites(out, blacks=a * 0.25, whites=a * 0.10)
    out = apply_vibrance(out, a * 0.20)
    return out


def apply_vignette(rgb01: ArrayF, amount: float, midpoint: float = 0.5) -> ArrayF:
    a = float(amount)
    if a == 0.0:
        return rgb01

    h, w = rgb01.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx / max(w - 1, 1)) * 2.0 - 1.0
    y = (yy / max(h - 1, 1)) * 2.0 - 1.0

    # Aspect-correct radius.
    aspect = w / max(h, 1)
    x = x * aspect
    r = np.sqrt(x * x + y * y)
    r = r / np.sqrt(aspect * aspect + 1.0)

    m = float(np.clip(midpoint, 0.0, 1.0))
    v = _smoothstep(m, 1.0, r)
    # a>0 darkens edges, a<0 lightens edges.
    factor = 1.0 - a * v
    return rgb01 * factor[..., None]


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
    blacks: float = 0.0
    whites: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    hue_degrees: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    shadows: float = 0.0
    highlights: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vignette: float = 0.0
    vignette_midpoint: float = 0.5
    pilgram_filter: str | None = None
    levels_black: float = 0.0
    levels_white: float = 1.0
    levels_gamma: float = 1.0
    curve_points: tuple[tuple[float, float], ...] | None = None
    curve_points_r: tuple[tuple[float, float], ...] | None = None
    curve_points_g: tuple[tuple[float, float], ...] | None = None
    curve_points_b: tuple[tuple[float, float], ...] | None = None
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
    if params.blacks or params.whites:
        out = apply_blacks_whites(out, params.blacks, params.whites)
    if params.temperature or params.tint:
        out = apply_temperature_tint(out, params.temperature, params.tint)
    if params.shadows or params.highlights:
        out = apply_shadows_highlights(out, params.shadows, params.highlights)
    if params.clarity:
        out = apply_midtone_contrast(out, params.clarity)
    if params.dehaze:
        out = apply_dehaze(out, params.dehaze)
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
    if params.curve_points_r is not None or params.curve_points_g is not None or params.curve_points_b is not None:
        out = apply_curve_rgb(out, params.curve_points_r, params.curve_points_g, params.curve_points_b)
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
    if params.vignette:
        out = apply_vignette(out, params.vignette, params.vignette_midpoint)
    if params.pilgram_filter:
        out = _apply_pilgram_filter(out, params.pilgram_filter)
    return np.clip(out, 0.0, 1.0)


def process_rgb8(rgb8: np.ndarray, params: FilterParams, strength: float) -> np.ndarray:
    rgb01 = _to_float01(rgb8)
    filtered01 = apply_params(rgb01, params)
    mixed01 = blend(rgb01, filtered01, strength)
    return _to_uint8(mixed01)


def process_rgb8_stack(rgb8: np.ndarray, params_stack: Sequence[FilterParams], strength: float) -> np.ndarray:
    rgb01 = _to_float01(rgb8)
    out01 = rgb01
    for params in params_stack:
        out01 = apply_params(out01, params)
    mixed01 = blend(rgb01, out01, strength)
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
