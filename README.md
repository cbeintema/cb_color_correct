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

### Windows image context menu

After running the app once, double-click `install_context_menu.bat` to add
**Open in CB Color Correct** for PNG, JPEG, TIFF, BMP, and WebP images.
On Windows 11, look under **Show more options** if necessary.
Each invocation opens the selected image in a new app window.
This installs only for your Windows account and does not require administrator access
or change the default image viewer. Keep this project folder in place; rerun the
installer if you move it. To uninstall, run `install_context_menu.bat --remove`.

You can also run `run.bat "C:\path\to\image.jpg"`.

### Editing features

- Load an image
- Pick a preset (Instagram-like looks)
- Load a .cube LUT and use it like a preset
- Adjust a strength slider (blends original → filtered)
- Draw one or more circular censor regions with a shared Gaussian blur amount
- Save the filtered image
- Upscale the current edited image through ComfyUI using NVIDIA RTX or SeedVR
- Create a preview package containing a censored image and a ZIP of the uncensored upscale
- Write an editable image description with a local LM Studio vision model, then copy it into a DeviantArt post
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

## Write a description for DeviantArt

1. Start LM Studio 0.4.0 or newer and enable its local server at `http://127.0.0.1:1234`. Alternatively, run `lms server start --port 1234 --bind 127.0.0.1` with LM Studio available.
2. Ensure the vision model and its companion `mmproj` file are installed together. The default is the downloaded DavidAU Qwen3.5 9B Defiant Fable Q4_K_S model. LM Studio 0.4.24 and this exact model/projector were tested successfully.
3. Load and edit an image in CB Color Correct. Expand the right **Tools** sidebar (or press **M**) and select its **Description** tab, beside **Modifications**. Click **Write description**. The model loads on demand; the completed description appears in the text field.
4. Edit the wording if needed, click **Copy**, and paste it into your DeviantArt post manually.

**Description settings** provides the local server address, **Refresh models**, a vision-model selector, **Load model**, context size, optional API token, and editable writing instructions. The default prompt requests 80–150 words of gallery prose grounded in the image. Model availability comes from LM Studio; the app does not download models. No new Python dependencies are needed.

The default source is **Edited artwork (without censor blur)**. Choose **Censored preview** to include the current blur regions. The request includes the current color edits, never the split comparison, zoom crop, or UI overlays. A metadata-free PNG is prepared in memory, capped at 1,280 pixels on its longest edge. Image and text go only to the configured localhost service; generated text and API tokens are not saved in app settings, and API chat storage is disabled. Your source file is unchanged.

You can cancel a request. Changing the image or edits cancels/discards old results; an existing description is marked out of date. Loading a new image clears the old description. Generation displays the complete result at once, preserving the previous draft if a retry fails. Closing the window during a request cancels it and waits without blocking the UI. Cancellation aborts the client connection; LM Studio may still finish a server operation, especially model loading. Check its activity before starting another GPU job.

Description and upscale jobs cannot start simultaneously in this app. Loaded models can still occupy GPU memory while idle: use **Unload app's model** before upscaling if needed. This only unloads the selected instance loaded by this app in this session. Models loaded elsewhere, after a cancelled load with no returned instance ID, or in a previous app session should be managed directly in LM Studio. The feature does not terminate ComfyUI or LM Studio.

On the tested RTX 3080 Ti (12 GB), a synthetic image request with 8,192 context tokens took about 13 seconds including loading and reached approximately 10.1 GiB total sampled GPU memory usage. This is one test, not a performance guarantee. For out-of-memory errors, unload other GPU models or adjust GPU offload in LM Studio. Context settings apply only to newly loaded instances. For missing vision support, check the companion projector and LM Studio runtime; for connection errors, check the local server; for authentication errors, enter the API token in Description settings.

Automated tests: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
Optional live check: `.venv\Scripts\python.exe tests/manual_description_smoke.py`. This uses a synthetic image and your local model, records screenshots/timing under ignored `.description-qa/`, and unloads the instance it loaded.
