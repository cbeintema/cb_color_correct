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
- Save the filtered image
- Includes an "Instagram" preset category powered by `pilgram2`

## Notes

- Preview is downscaled for responsiveness; saving applies the same preset to full resolution.
- This app focuses on color/style (no masking, selection, healing, etc.).
- LUTs: supports common .cube 1D and 3D LUTs (trilinear interpolation for 3D).
