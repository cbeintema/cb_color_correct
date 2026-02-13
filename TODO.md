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

### Recently completed (2026-02-12)

- Curves toggle bug fix: disabling Curves no longer discards curve data; re-enabling reapplies existing curve state immediately.
- Favorites list stability fixes:
  - resolved persistent preset row spacing growth on favorite toggle,
  - moved favorites star to text column interaction,
  - enforced stable row heights.
- Favorites view behavior:
  - when `Fav only` is enabled, visible categories auto-expand,
  - previous non-favorites expand/collapse state is restored when exiting `Fav only`.
- Preview interaction upgrades:
  - zoom controls (`+`, `-`, `Fit`, `Actual Size`),
  - panning via middle-drag and Space+drag,
  - macOS pinch zoom and Cmd/Ctrl+wheel zoom,
  - draggable split preview divider.
- Toolbar UX updates:
  - `Adjustments` renamed to `Modifications`,
  - `M` toggles modifications panel,
  - undo/redo buttons added and grouped separately from zoom controls,
  - split preview control moved to centered position.
- Undo/redo improvements:
  - history stack added for filter adjustments,
  - shortcut handling strengthened for macOS (`Cmd+Z`, `Cmd+Shift+Z`) with fallbacks.
- File workflow improvements:
  - shortcuts for open/save (`Cmd/Ctrl+O`, `Cmd/Ctrl+S`),
  - open dialog remembers last used image folder.
- Zoom hotkeys added:
  - `Cmd/Ctrl +` zoom in,
  - `Cmd/Ctrl -` zoom out,
  - `Cmd/Ctrl+A` actual size,
  - `Cmd/Ctrl+F` fit.

- Basic preset browser with categories
- LUT load (.cube 1D/3D)
- Adjustment sliders: Exposure/Brightness/Contrast/Hue/Saturation/Vibrance + Levels
- Curves editor widget (master curve, draggable points + Reset + S-Curve)
- Temperature / Tint (WB), Shadows / Highlights, Blacks / Whites sliders
- Split Tone controls (shadows/highlights colors + amount/balance)
- Effects: Clarity (midtone contrast), Dehaze (approx), Vignette
