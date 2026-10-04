# STL to LEGO Converter

## Overview
This is a Flask web application that converts STL 3D mesh files into layered LEGO brick models. Users can upload STL files and receive layered images, PDF assembly instructions, and a 3D STL of the brick assembly.

## Features
- Upload STL files and choose processing options
- Select number of layers (10-100) and slicing direction (X, Y, Z)
- Make model hollow (shell extraction)
- Overlay previous layer for visual context
- Color bricks by shape
- Generate PDF assembly instructions
- Remove hanging bricks (for non-hollow models)

## Project Structure
- `app.py` - Main Flask application with all routes and processing logic
- `templates/` - HTML templates (index.html, results.html, history.html)
- `static/results/` - Generated outputs (images, PDFs, STL files)
- `uploads/` - Uploaded STL files
- `examples/` - Example cURL scripts

## Running the App
The app runs on port 5000 with the following command:
```bash
LD_LIBRARY_PATH="/nix/store/bmi5znnqk4kg2grkrhk6py0irc8phf6l-gcc-14.2.1.20250322-lib/lib:$LD_LIBRARY_PATH" python app.py
```

## Key Dependencies
- Flask - Web framework
- brickalize - Core LEGO brick conversion library
- open3d-cpu - 3D geometry processing (CPU-only version)
- trimesh - 3D mesh handling
- Pillow - Image processing
- fpdf2 - PDF generation
- scipy/numpy - Numerical computing

## Routes
- `/` - Upload form
- `/upload` - POST file and options
- `/history` - List previous results
- `/history/<result_folder>` - View specific result
- `/download/<filename>` - Download generated STL
- `/download_pdf/<filename>` - Download instruction PDF

## Deployment
Configured for autoscale deployment using gunicorn.
