# STL to LEGO Converter

Convert STL mesh files into layered LEGO brick models with images, optional PDF instructions, and a 3D STL of the brick assembly.

## Features
- Upload `.stl` and choose processing options
  - Number of layers and slicing direction (`X`, `Y`, `Z`)
  - Make model hollow (shell extraction)
  - Overlay previous layer for visual context
  - Color bricks by shape
  - Generate PDF assembly instructions
  - Remove hanging bricks (non-hollow): ensures each brick has support below
- Outputs
  - Per-layer PNGs with grid and brick outlines
  - Optional PDF instruction booklet
  - Brick-assembled 3D model `.stl`

## Requirements
- Python 3.10+
- pip
- Git (optional; no longer required since `brickalize` is pinned from PyPI)

Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- On Windows, SciPy may need build tools. If installation fails, install "Microsoft C++ Build Tools" (C++ workload) and try again.
- On macOS, install Xcode Command Line Tools: `xcode-select --install`.
- Ensure Git is installed and available on PATH so pip can fetch `brickalize` from GitHub.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` and upload an STL.

### Endpoints
- `/` upload form
- `/upload` POST file and options
- `/history` list previous results
- `/history/<result_folder>` view a specific result
- `/results/<result_folder>/<filename>` serve result assets
- `/download/<filename>` download generated STL
- `/download_pdf/<filename>` download instruction PDF

## Usage Notes
- Images and models are written under `static/results/<filename_without_ext>/`.
- Uploaded files are saved to `uploads/`.
- If "Create PDF assembly instructions" is checked, a PDF is written to `static/results/<filename>_instructions.pdf` and a download button appears on the results page.
- If "Color bricks by shape" is enabled, layer images include a legend plus dark outlines for easier identification. Default images also include dark outlines.
- "Remove Hanging Bricks" removes unsupported bricks in non-hollow mode so every brick rests on at least one brick below.

Fonts: the app will try to use `arial.ttf` and `arialbd.ttf` for labels; if unavailable it falls back to default fonts.

## Example: cURL Upload

You can POST to `/upload` with multipart form-data:

```bash
curl -X POST http://127.0.0.1:5000/upload \
  -F "file=@uploads/StanfordBunny_fixed.stl" \
  -F "grid_voxel_count=50" \
  -F "grid_direction=z" \
  -F "extract_shell=" \
  -F "overlay_layers=" \
  -F "generate_pdf=" \
  -F "color_by_shape=" \
  -F "remove_hanging_bricks="
```

Checkbox options are enabled by including the field (any value). Omit a field to disable it.

## Project Layout
- `app.py`: Flask app and processing pipeline
- `templates/`: HTML templates
- `static/results/`: Outputs (images, PDF, STL)
- `uploads/`: Uploaded STL files
- `examples/`: Example scripts and requests

## Troubleshooting
- PDF link 404: ensure "Create PDF assembly instructions" was checked and that `static/results/<file>_instructions.pdf` exists. Reload the results page.
- Large STLs: increase layers gradually; too many layers can be slow/memory-intensive.

## License
This project is provided as-is by the repository owner. Add a license if needed.