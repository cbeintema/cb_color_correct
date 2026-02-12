from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np


@dataclass(frozen=True)
class Lut1D:
    title: str | None
    size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    table: np.ndarray  # shape (size, 3), float32 in 0..1 typically


@dataclass(frozen=True)
class Lut3D:
    title: str | None
    size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    table: np.ndarray  # shape (size, size, size, 3) with axes (r, g, b)


CubeLut = Union[Lut1D, Lut3D]


class CubeParseError(ValueError):
    pass


def _iter_data_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            yield line


def load_cube(path: str | Path) -> CubeLut:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    title: str | None = None
    lut_1d_size: int | None = None
    lut_3d_size: int | None = None
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    data: list[tuple[float, float, float]] = []

    for line in _iter_data_lines(text):
        parts = line.split()
        if not parts:
            continue

        head = parts[0].upper()
        if head == "TITLE":
            # TITLE "My LUT"
            rest = line[len(parts[0]) :].strip()
            if rest.startswith('"') and rest.endswith('"') and len(rest) >= 2:
                title = rest[1:-1]
            else:
                title = rest.strip('"') or None
            continue

        if head == "LUT_1D_SIZE":
            if len(parts) != 2:
                raise CubeParseError(f"Invalid LUT_1D_SIZE line: {line}")
            lut_1d_size = int(parts[1])
            continue

        if head == "LUT_3D_SIZE":
            if len(parts) != 2:
                raise CubeParseError(f"Invalid LUT_3D_SIZE line: {line}")
            lut_3d_size = int(parts[1])
            continue

        if head == "DOMAIN_MIN":
            if len(parts) != 4:
                raise CubeParseError(f"Invalid DOMAIN_MIN line: {line}")
            domain_min = (float(parts[1]), float(parts[2]), float(parts[3]))
            continue

        if head == "DOMAIN_MAX":
            if len(parts) != 4:
                raise CubeParseError(f"Invalid DOMAIN_MAX line: {line}")
            domain_max = (float(parts[1]), float(parts[2]), float(parts[3]))
            continue

        # Data row: 3 floats
        if len(parts) >= 3:
            try:
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as e:
                raise CubeParseError(f"Invalid data row: {line}") from e
            data.append((r, g, b))
            continue

    if lut_1d_size and lut_3d_size:
        raise CubeParseError(".cube contains both LUT_1D_SIZE and LUT_3D_SIZE")
    if not lut_1d_size and not lut_3d_size:
        raise CubeParseError("Missing LUT_1D_SIZE / LUT_3D_SIZE")

    if lut_1d_size:
        size = lut_1d_size
        expected = size
        if len(data) < expected:
            raise CubeParseError(f"Not enough LUT rows: expected {expected}, got {len(data)}")
        table = np.asarray(data[:expected], dtype=np.float32)
        return Lut1D(title=title, size=size, domain_min=domain_min, domain_max=domain_max, table=table)

    size = int(lut_3d_size)
    expected = size * size * size
    if len(data) < expected:
        raise CubeParseError(f"Not enough LUT rows: expected {expected}, got {len(data)}")
    flat = np.asarray(data[:expected], dtype=np.float32)

    # Most common .cube ordering is R changes fastest, then G, then B.
    # Reshape to (b, g, r, 3) then transpose to (r, g, b, 3).
    table_bgr = flat.reshape((size, size, size, 3))
    table_rgb = np.transpose(table_bgr, (2, 1, 0, 3)).copy()
    return Lut3D(title=title, size=size, domain_min=domain_min, domain_max=domain_max, table=table_rgb)


def apply_lut(rgb01: np.ndarray, lut: CubeLut) -> np.ndarray:
    if isinstance(lut, Lut1D):
        return apply_lut_1d(rgb01, lut)
    return apply_lut_3d(rgb01, lut)


def _domain_normalize(rgb01: np.ndarray, domain_min: tuple[float, float, float], domain_max: tuple[float, float, float]) -> np.ndarray:
    dmin = np.array(domain_min, dtype=np.float32)
    dmax = np.array(domain_max, dtype=np.float32)
    denom = np.maximum(dmax - dmin, 1e-6)
    return np.clip((rgb01 - dmin) / denom, 0.0, 1.0)


def apply_lut_1d(rgb01: np.ndarray, lut: Lut1D) -> np.ndarray:
    x = _domain_normalize(rgb01, lut.domain_min, lut.domain_max)

    n = lut.size
    pos = x * (n - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, n - 1)
    t = (pos - i0).astype(np.float32)

    # For 1D cube LUTs, each row is an RGB triplet. Apply per channel with corresponding column.
    table = lut.table

    out = np.empty_like(rgb01, dtype=np.float32)
    for c in range(3):
        v0 = table[i0[..., c], c]
        v1 = table[i1[..., c], c]
        out[..., c] = v0 * (1.0 - t[..., c]) + v1 * t[..., c]

    return np.clip(out, 0.0, 1.0)


def apply_lut_3d(rgb01: np.ndarray, lut: Lut3D) -> np.ndarray:
    x = _domain_normalize(rgb01, lut.domain_min, lut.domain_max)

    n = lut.size
    pos = x * (n - 1)

    r0 = np.floor(pos[..., 0]).astype(np.int32)
    g0 = np.floor(pos[..., 1]).astype(np.int32)
    b0 = np.floor(pos[..., 2]).astype(np.int32)

    r1 = np.clip(r0 + 1, 0, n - 1)
    g1 = np.clip(g0 + 1, 0, n - 1)
    b1 = np.clip(b0 + 1, 0, n - 1)

    tr = (pos[..., 0] - r0).astype(np.float32)[..., None]
    tg = (pos[..., 1] - g0).astype(np.float32)[..., None]
    tb = (pos[..., 2] - b0).astype(np.float32)[..., None]

    t = lut.table

    c000 = t[r0, g0, b0]
    c100 = t[r1, g0, b0]
    c010 = t[r0, g1, b0]
    c110 = t[r1, g1, b0]
    c001 = t[r0, g0, b1]
    c101 = t[r1, g0, b1]
    c011 = t[r0, g1, b1]
    c111 = t[r1, g1, b1]

    c00 = c000 * (1.0 - tr) + c100 * tr
    c10 = c010 * (1.0 - tr) + c110 * tr
    c01 = c001 * (1.0 - tr) + c101 * tr
    c11 = c011 * (1.0 - tr) + c111 * tr

    c0 = c00 * (1.0 - tg) + c10 * tg
    c1 = c01 * (1.0 - tg) + c11 * tg

    out = c0 * (1.0 - tb) + c1 * tb
    return np.clip(out.astype(np.float32), 0.0, 1.0)
