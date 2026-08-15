# CB Color Correct

Simple desktop image color-correction preset tool (PySide6).

## Run

macOS: double-click `run.command` (or run `./run.command`)

Windows: double-click `run.bat`

These scripts will create `.venv` and install `requirements.txt` on first run.

Manual:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

## What it does

- Load an image
- Pick a preset (Instagram-like looks)
- Load a .cube LUT and use it like a preset
- Adjust a strength slider (blends original → filtered)
- Draw one or more circular censor regions with a shared Gaussian blur amount
- Save the filtered image
- Upscale the current edited image through ComfyUI using NVIDIA RTX or SeedVR
- Create a preview package containing a censored image and a ZIP of the uncensored upscale
- Includes an "Instagram" preset category powered by `pilgram2`

## Notes

- Preview is downscaled for responsiveness; saving applies the same preset to full resolution.
- Censoring is manual, non-destructive, and applies to one loaded image at a time; batch processing does not use censor regions.
- LUTs: supports common .cube 1D and 3D LUTs (trilinear interpolation for 3D).
- Upscaling is available in the `Upscale` tab. It sends a temporary PNG containing the current full-resolution color-corrected image to ComfyUI; censor regions are excluded from the upscale source. The source image is not modified.
- `Upscale and Zip` requires at least one censor circle. For an input such as `portrait.jpg`, it creates `portrait/portrait_censored.jpg` and `portrait/portrait.zip`; the ZIP contains only `portrait_upscaled.png`. An optional Package Name in the Upscale tab replaces `portrait` in the package folder and all three generated file names.
- Generated preview and upscale images are re-encoded without embedded metadata. The generated ZIP has no optional comments, timestamps, extra fields, or filesystem attributes.
- ComfyUI must have the NVIDIA RTX Video Super Resolution node or the SeedVR2 nodes and required models installed, depending on the selected engine. Use the default `http://127.0.0.1:8000` URL. The app auto-fills the standard ComfyUI installation paths used by `image_cleaner`; otherwise configure the Python executable and `main.py` to launch a headless server from the tab. Headless startup output is written beside ComfyUI's `main.py` in `cb_color_correct_comfyui.log`.
- Upscaling requires the `requests` and `websocket-client` packages included in `requirements.txt`.
