# TODO

This file tracks larger follow-ups for CB Color Correct.

## In progress / next

### Curves editor widget

Goal: expose curves as a real widget (not just preset-defined points).

- Implement a simple curve editor (QGraphicsView) with:
  - 4–6 draggable control points in 0..1 space
  - clamped endpoints at (0,0) and (1,1)
  - optional reset + "S-curve" presets
- Convert control points to `FilterParams.curve_points` and apply live.
- Optional: per-channel curves later (RGB separate). Start with one master curve.

### More color tools

- HSL: currently we have Hue + Saturation and "Brightness" as a rough luminance control.
- Consider adding:
  - highlights/shadows (simple luma masks)
  - temperature/tint (approx via channel multipliers)

### Preset save/load improvements

Current behavior:
- Save writes JSON with both `base` and `adjust`.
- Load restores both base + adjustments.

Potential improvements:
- Store the currently-selected preset name/reference instead of embedding full base params.
- Optional: remember LUT source file path (if safe) so it can be reloaded.
- Add a dedicated "User" preset management section (rename/delete).

### Performance

- Apply is debounced to 25ms; still CPU-based.
- If images are large, consider:
  - running filter in a worker thread (QtConcurrent / QThread)
  - caching the last processed preview for slider drags

## Completed

- Basic preset browser with categories
- LUT load (.cube 1D/3D)
- Adjustment sliders: Exposure/Brightness/Contrast/Hue/Saturation/Vibrance + Levels
