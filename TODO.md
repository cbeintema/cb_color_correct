# TODO

This file tracks larger follow-ups for CB Color Correct.

## In progress / next

### Per-channel curves (optional)

- Implemented Master + per-channel (R/G/B) curves via a Channel selector in the Curves panel.
- Optional next: add "link" mode (edit Master and propagate to RGB) and/or show all curves overlayed.

### More color tools

- HSL: currently we have Hue + Saturation and "Brightness" as a rough luminance control.
- Consider adding:
  - split-tone controls (expose shadows/highlights color + amount)
  - optional: clarity/dehaze approximation (mid-tone contrast)

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
- Curves editor widget (master curve, draggable points + Reset + S-Curve)
- Temperature / Tint (WB), Shadows / Highlights, Blacks / Whites sliders
- Split Tone controls (shadows/highlights colors + amount/balance)
- Effects: Clarity (midtone contrast), Dehaze (approx), Vignette
