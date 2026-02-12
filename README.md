# CB Color Correct

Simple desktop image color-correction preset tool (PySide6).

## Run

```bash
/Users/christopherrbeintema/cb_color_correct/.venv/bin/python main.py
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
