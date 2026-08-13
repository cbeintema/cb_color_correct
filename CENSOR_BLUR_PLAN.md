# Circular Censor Blur Plan

## Status

Implemented in CB Color Correct.

## Goal

Add a non-destructive, single-image censoring tool. The user enters Censor mode, draws one or more circular regions over the preview, and exports a copy with Gaussian blur applied only inside those regions.

The original file remains unchanged. This feature is intentionally excluded from the existing batch workflow because every image needs its own manually placed regions.

## Version 1 Scope

- One loaded image at a time.
- Click-drag from a circle center to its edge to create a circular censor region.
- Multiple regions per image.
- One global Gaussian blur amount shared by all regions.
- Visible outlines and a subtle translucent fill while editing.
- Undo, redo, remove-last, and clear-all actions.
- Preview rendering at the current zoom level and correct full-resolution output on Save As.

Not in Version 1:

- Automatic face/object detection.
- Elliptical, freehand, or rectangular masks.
- Dragging or resizing an existing region.
- Applying the same regions to a batch of images.
- Editing animated images or video frames.

## User Experience

Add a compact Censor section to the preview toolbar in `main.py`:

- Checkable `Censor` button: enables drawing mode and shows a crosshair cursor.
- `Blur` spin box or slider: global Gaussian radius in full-resolution pixels. A practical initial range is 1 to 100 pixels with a default of 24.
- `Remove Last` button: removes the most recently drawn region.
- `Clear` button: removes all censor regions after confirmation only when more than one exists.

Interaction rules:

1. With Censor mode enabled, left-drag sets the center at mouse-down and the radius at mouse-up.
2. A very small drag is ignored rather than creating an unusable region.
3. `Escape` cancels an in-progress drag.
4. Space-drag and middle-drag retain the existing pan behavior.
5. Disable Split Preview while Censor mode is active, because both features need left-drag input on the image.
6. Re-render the preview after a completed region or blur-amount change; do not re-render for every mouse-move event.

## Data Model

Represent each circle using image-relative coordinates so zoom, window size, and the 1600-pixel preview cap cannot affect its exported position:

```python
@dataclass(frozen=True)
class CensorCircle:
    center_x: float  # 0..1, relative to image width
    center_y: float  # 0..1, relative to image height
    radius: float    # relative to image width
```

Keep the blur amount as a separate `int` in full-resolution pixels. To render a downscaled preview, multiply it by:

$$
\frac{\text{preview width}}{\text{original width}}
$$

The screen-to-image mapping is straightforward because the preview label is sized exactly to the displayed pixmap: divide local mouse coordinates by the label width and height. Store the radius relative to label width.

## Rendering and Export

Create a small, pure helper module such as `cb_color_correct/censor.py` containing:

- `CensorCircle`
- `apply_censor_blur(rgb8, circles, blur_radius)`

`apply_censor_blur` should:

1. Receive an RGB `numpy.uint8` array.
2. Convert it to a Pillow image.
3. For each circle, crop a bounding rectangle expanded by at least three times the blur radius.
4. Blur that crop with `PIL.ImageFilter.GaussianBlur`.
5. Composite the blurred crop through a grayscale `ImageDraw.ellipse` mask so pixels outside the circle are unchanged.
6. Return an RGB `numpy.uint8` array.

Cropping avoids repeatedly blurring a full 4K or 8K image when only a small face or label needs censoring. It also supports circles with different sizes without adding a new dependency.

Apply the helper after `process_rgb8_stack` in both paths:

- `_RenderTask.run()` for the downscaled interactive preview, with scaled blur radius.
- `MainWindow._on_save()` for the full-resolution export, with the configured blur radius.

Applying censorship last ensures color adjustments cannot reduce the final blur effect.

## Canvas Integration

Replace the plain preview `QLabel` with a small `CensorImageLabel` subclass in `main.py`.

The subclass should:

- Keep the current scaled pixmap as its base image.
- Draw committed circles and the in-progress drag circle in `paintEvent` using `QPainter`.
- Emit normalized circle coordinates after a completed drag.
- Handle only left-drag while Censor mode is enabled and Space is not held.

Leave zoom and pan in `_PanScrollArea`. This keeps the existing viewport behavior intact and avoids mixing drawing state with the filter-rendering pixmaps. Existing split-divider handling can remain in the window event filter after Censor mode has disabled the split control.

## State and History

Extend `_HistoryState` to include:

- An immutable tuple of `CensorCircle` values.
- The global blur radius.

Update `_make_history_state`, equality comparison, and `_apply_history_state` so completed add/remove/clear and blur changes work with the existing undo/redo controls. Use immutable values so a later edit cannot mutate an already-recorded history entry.

When loading a new image, clear censor circles, restore the default blur radius, and reset undo/redo history. Censor state must never be passed into `_BatchTask`.

## Files Expected to Change

- `main.py`: UI controls, preview label subclass, state, history integration, coordinate mapping, preview task arguments, and export integration.
- `cb_color_correct/censor.py`: pure circle model and masked Gaussian-blur implementation.
- `README.md`: add the censor tool to the feature list and clarify that it is manual and single-image only.
- `tests/test_censor.py`: focused behavior tests using `unittest` so no new test dependency is required.

## Verification

Automated tests:

- No circles returns byte-identical pixels.
- Pixels outside a circle remain unchanged.
- Pixels inside a patterned circle are blurred.
- Circles partly outside an image are clipped safely.
- Normalized coordinates map to the expected full-resolution location.
- Multiple circles are applied and the batch code does not receive censor state.

Manual checks:

1. Load a large portrait and landscape image, draw circles at Fit and at 100% zoom, then save and verify placement in the exported image.
2. Pan with Space or middle mouse while Censor mode is both on and off.
3. Confirm Split Preview is unavailable only while drawing mode is active and works again afterward.
4. Confirm undo/redo, Remove Last, Clear, and Escape leave the preview and export state consistent.
5. Run `python -m unittest discover -s tests`, `python -m compileall main.py cb_color_correct`, and launch `python main.py` for a smoke test.