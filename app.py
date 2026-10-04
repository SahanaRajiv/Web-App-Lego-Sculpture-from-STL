from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import os
import time
import re
from werkzeug.utils import secure_filename
import shutil
from fpdf import FPDF, XPos, YPos
import trimesh
from PIL import Image, ImageDraw, ImageFont
import random
from collections import Counter
import requests
from datetime import datetime, timedelta
from google.cloud import storage

def numpy_label(input_array):
    """Pure numpy connected component labeling (4-connectivity for 2D)."""
    labeled = np.zeros_like(input_array, dtype=int)
    current_label = 0
    rows, cols = input_array.shape
    
    for i in range(rows):
        for j in range(cols):
            if input_array[i, j] > 0 and labeled[i, j] == 0:
                current_label += 1
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= rows or c < 0 or c >= cols:
                        continue
                    if input_array[r, c] > 0 and labeled[r, c] == 0:
                        labeled[r, c] = current_label
                        stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
    
    return labeled, current_label

from brickalize_bricks import Brick, BrickSet
from brickalize_model import BrickModel
from brickalize_converter import Brickalizer
from lightweight_visualizer import LightweightVisualizer as BrickModelVisualizer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPLIT_SIDECAR_ENDPOINT = "http://127.0.0.1:1106"

def get_storage_client():
    try:
        return storage.Client(
            credentials=None,
            project="",
        )
    except Exception:
        try:
            creds_config = {
                "audience": "replit",
                "subject_token_type": "access_token",
                "token_url": f"{REPLIT_SIDECAR_ENDPOINT}/token",
                "type": "external_account",
                "credential_source": {
                    "url": f"{REPLIT_SIDECAR_ENDPOINT}/credential",
                    "format": {
                        "type": "json",
                        "subject_token_field_name": "access_token",
                    },
                },
            }
            from google.auth import external_account
            credentials = external_account.Credentials.from_info(creds_config)
            return storage.Client(credentials=credentials, project="")
        except Exception as e:
            print(f"Failed to create storage client: {e}")
            return None

def get_object_storage_bucket():
    bucket_id = os.environ.get("DEFAULT_OBJECT_STORAGE_BUCKET_ID", "")
    if bucket_id:
        return bucket_id
    return os.environ.get("OBJECT_STORAGE_BUCKET", "")

def sign_object_url_via_sidecar(bucket_name: str, object_name: str, method: str = "GET", ttl_sec: int = 3600) -> str:
    expires_at = (datetime.utcnow() + timedelta(seconds=ttl_sec)).isoformat() + "Z"
    
    request_data = {
        "bucket_name": bucket_name,
        "object_name": object_name,
        "method": method,
        "expires_at": expires_at,
    }
    
    response = requests.post(
        f"{REPLIT_SIDECAR_ENDPOINT}/object-storage/signed-object-url",
        json=request_data,
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    
    if not response.ok:
        raise Exception(f"Failed to sign object URL: {response.status_code}")
    
    return response.json()["signed_url"]

def upload_to_object_storage(local_path: str, object_name: str) -> str:
    bucket_name = get_object_storage_bucket()
    if not bucket_name:
        print("No bucket configured")
        return None
    
    try:
        upload_url = sign_object_url_via_sidecar(bucket_name, object_name, method="PUT", ttl_sec=900)
        
        with open(local_path, 'rb') as f:
            content = f.read()
        
        ext = object_name.lower().split('.')[-1]
        content_types = {
            'stl': 'application/sla',
            'pdf': 'application/pdf',
            'png': 'image/png',
        }
        content_type = content_types.get(ext, 'application/octet-stream')
        
        response = requests.put(
            upload_url,
            data=content,
            headers={"Content-Type": content_type},
            timeout=60,
        )
        
        if response.ok:
            print(f"Uploaded {object_name} to object storage")
            return object_name
        else:
            print(f"Upload failed: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Object storage upload failed: {e}")
    
    return None

def get_object_download_url(object_name: str, ttl_sec: int = 3600) -> str:
    bucket_name = get_object_storage_bucket()
    if not bucket_name:
        return None
    
    try:
        return sign_object_url_via_sidecar(bucket_name, object_name, method="GET", ttl_sec=ttl_sec)
    except Exception as e:
        print(f"Sidecar signing failed: {e}, trying GCS client")
        try:
            client = get_storage_client()
            if client:
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(object_name)
                if blob.exists():
                    url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(seconds=ttl_sec),
                        method="GET",
                    )
                    return url
        except Exception as e2:
            print(f"GCS client signing also failed: {e2}")
        return None

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'stl'}
STARTER_MODEL_LABELS = {
    'Rubber_Duck.stl': 'Rubber Duck (starter model)',
    'StanfordBunny_fixed.stl': 'Stanford Bunny (starter model)',
}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def draw_grid_on_image(image_path, num_cells_x, num_cells_y, pixels_per_stud=20):
    """Draws a grid on the image and adds axis numbers, ensuring a square grid."""
    margin = 30

    # Calculate the target dimensions for a square grid
    target_width = num_cells_x * pixels_per_stud
    target_height = num_cells_y * pixels_per_stud
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        
        # Resize the image to the target dimensions to enforce a square grid
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Create a new canvas with margins for axis labels.
        canvas_width = target_width + margin
        canvas_height = target_height + margin
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        
        # Paste the corrected image onto the canvas, offset by the margin.
        canvas.paste(img, (margin, margin))

        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw grid lines over the pasted image area.
        for i in range(num_cells_x + 1):
            x = margin + (i * pixels_per_stud)
            draw.line([(x, margin), (x, target_height + margin)], fill="lightgrey", width=1)
        for i in range(num_cells_y + 1):
            y = margin + (i * pixels_per_stud)
            draw.line([(margin, y), (target_width + margin, y)], fill="lightgrey", width=1)

        # Draw X-axis numbers in the top margin.
        for i in range(num_cells_x):
            x = margin + (i * pixels_per_stud) + (pixels_per_stud / 2)
            draw.text((x, margin / 2), str(i), fill="black", font=font, anchor="mm")

        # Draw Y-axis numbers in the left margin.
        for i in range(num_cells_y):
            y = margin + (i * pixels_per_stud) + (pixels_per_stud / 2)
            draw.text((margin / 2, y), str(i), fill="black", font=font, anchor="mm")
            
        canvas.save(image_path)


def draw_brick_outlines_on_image(image_path, layer_num, brick_model, pixels_per_stud=20):
    """Overlays dark outlines for each brick of a given layer onto an existing grid image.
    Assumes the image has been resized to pixels_per_stud with a 30px margin (same as draw_grid_on_image)."""
    margin = 30
    if not brick_model.layers or layer_num not in brick_model.layers:
        return

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        outline_color = (20, 20, 20)
        outline_width = 2
        
        for brick in brick_model.layers[layer_num]:
            x0 = margin + (brick['position'][0] * pixels_per_stud)
            y0 = margin + (brick['position'][1] * pixels_per_stud)
            x1 = x0 + (brick['size'][0] * pixels_per_stud)
            y1 = y0 + (brick['size'][1] * pixels_per_stud)
            draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=outline_width)
        
        img.save(image_path)


def generate_custom_layer_images(brick_model, image_folder, shape_colors, pixels_per_stud=20):
    """Generates layer images from scratch with color-coded bricks and a legend on a single canvas."""
    
    # --- Data Preparation ---
    running_total_counts = Counter()
    total_brick_counts = Counter()
    
    parsed_layers = {}
    if brick_model.layers:
        for z, layer_bricks in brick_model.layers.items():
            if z not in parsed_layers:
                parsed_layers[z] = []
            for brick in layer_bricks:
                width, height = brick['size']
                shape = tuple(sorted((width, height)))
                total_brick_counts[shape] += 1
                
                parsed_layers[z].append({
                    'width': width,
                    'height': height,
                    'x': brick['position'][0],
                    'y': brick['position'][1],
                    'shape': shape
                })

    legend_entries = sorted(total_brick_counts.keys())

    # --- Layout and Font Setup ---
    margin = 30
    legend_padding = 10
    max_x = brick_model.size[0]
    max_y = brick_model.size[1]
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        header_font = ImageFont.truetype("arialbd.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        header_font = font

    # --- Calculate Legend Dimensions ---
    legend_entries = sorted(total_brick_counts.keys())
    entry_height = 25
    header_height = 30
    color_swatch_size = 20
    col_widths = {"color": color_swatch_size + 10, "size": 70, "layer_count": 80, "total_count": 80}
    legend_width = sum(col_widths.values())
    legend_height = header_height + len(legend_entries) * entry_height
    
    # --- Generate Image for Each Layer ---
    for layer_num in range(brick_model.size[2]):
        
        # --- Create Canvas ---
        brick_area_width = max_x * pixels_per_stud
        brick_area_height = max_y * pixels_per_stud
        
        canvas_width = margin + brick_area_width + legend_padding + legend_width
        canvas_height = max(margin + brick_area_height, legend_height + margin)
        
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        # --- Draw Bricks ---
        layer_brick_counts = Counter()
        if layer_num in parsed_layers:
            for brick in parsed_layers[layer_num]:
                running_total_counts[brick['shape']] += 1
                layer_brick_counts[brick['shape']] += 1
                
                x0 = margin + (brick['x'] * pixels_per_stud)
                y0 = margin + (brick['y'] * pixels_per_stud)
                x1 = x0 + (brick['width'] * pixels_per_stud)
                y1 = y0 + (brick['height'] * pixels_per_stud)
                
                color = shape_colors[brick['shape']]
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(20, 20, 20), width=2)

        # --- Draw Grid and Axis Labels ---
        for i in range(max_x + 1):
            x = margin + (i * pixels_per_stud)
            draw.line([(x, margin), (x, margin + brick_area_height)], fill="lightgrey", width=1)
        for i in range(max_y + 1):
            y = margin + (i * pixels_per_stud)
            draw.line([(margin, y), (margin + brick_area_width, y)], fill="lightgrey", width=1)

        for i in range(max_x):
            x = margin + (i * pixels_per_stud) + (pixels_per_stud / 2)
            draw.text((x, margin / 2), str(i), fill="black", font=font, anchor="mm")
        for i in range(max_y):
            y = margin + (i * pixels_per_stud) + (pixels_per_stud / 2)
            draw.text((margin / 2, y), str(i), fill="black", font=font, anchor="mm")

        # --- Draw Legend ---
        legend_x_start = margin + brick_area_width + legend_padding
        
        # Header
        x = legend_x_start
        headers = {"color": "Color", "size": "Size", "layer_count": "Layer #", "total_count": "Total #"}
        for col, text in headers.items():
            draw.text((x + col_widths[col]/2, margin + header_height/2), text, font=header_font, fill="black", anchor="mm")
            x += col_widths[col]
        draw.line([(legend_x_start, margin + header_height - 1), (legend_x_start + legend_width, margin + header_height - 1)], fill="black", width=1)

        # Entries
        y = margin + header_height
        for shape in legend_entries:
            x = legend_x_start
            
            color = shape_colors[shape]
            draw.rectangle([x + 5, y + (entry_height - color_swatch_size)/2, x + 5 + color_swatch_size, y + (entry_height + color_swatch_size)/2], fill=color, outline="black")
            x += col_widths["color"]
            
            draw.text((x + col_widths["size"]/2, y + entry_height/2), f"{shape[0]}x{shape[1]}", font=font, fill="black", anchor="mm")
            x += col_widths["size"]

            count = layer_brick_counts.get(shape, 0)
            draw.text((x + col_widths["layer_count"]/2, y + entry_height/2), str(count), font=font, fill="black", anchor="mm")
            x += col_widths["layer_count"]
            
            count = running_total_counts.get(shape, 0)
            draw.text((x + col_widths["total_count"]/2, y + entry_height/2), str(count), font=font, fill="black", anchor="mm")
            
            y += entry_height

        # --- Save Final Image ---
        image_path = os.path.join(image_folder, f"layer_{layer_num}.png")
        canvas.save(image_path)


def create_pdf_instructions(image_files, image_folder, pdf_path, brick_model, shape_colors):
    """Creates a PDF with assembly instructions."""
    pdf = FPDF(orientation='L')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Create Grand Total Legend Page ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Total Bricks Required", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_y(30)
    
    total_brick_counts = Counter()
    if brick_model.layers:
        for layer_num in brick_model.layers:
            for brick in brick_model.layers[layer_num]:
                shape = tuple(sorted(brick['size']))
                total_brick_counts[shape] += 1

    legend_entries = sorted(total_brick_counts.keys())
    
    pdf.set_font("Helvetica", "B", 12)
    col_widths = {"color": 20, "size": 40, "total": 40}
    
    pdf.cell(col_widths["color"], 10, "Color", border=1, align="C")
    pdf.cell(col_widths["size"], 10, "Size", border=1, align="C")
    pdf.cell(col_widths["total"], 10, "Total Bricks", border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 12)
    for shape in legend_entries:
        color = shape_colors.get(shape, (0,0,0))
        pdf.set_fill_color(color[0], color[1], color[2])
        pdf.cell(col_widths["color"], 10, "", border=1, fill=True)
        pdf.cell(col_widths["size"], 10, f"{shape[0]}x{shape[1]}", border=1, align="C")
        pdf.cell(col_widths["total"], 10, str(total_brick_counts[shape]), border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # --- Add Layer Pages ---
    for i, image_file in enumerate(image_files):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Layer {i}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        img_path = os.path.join(image_folder, image_file)
        pdf.image(img_path, x=10, y=25, w=277)
        
    pdf.output(pdf_path)

def overlay_image_layers(image_files, image_folder):
    """Overlays each layer on a lighter version of the one immediately below it."""
    # This variable will hold a clean copy of the previous layer's image,
    # to avoid the cascading effect of compositing on an already-composited image.
    previous_layer_image = None

    for i in range(len(image_files)):
        current_img_path = os.path.join(image_folder, image_files[i])
        
        with Image.open(current_img_path) as current_img_with_grid:
            
            # Only perform the overlay for layers after the first one.
            if i > 0 and previous_layer_image:
                # --- Process previous layer ---
                # Convert the clean previous layer image to a light, semi-transparent grey.
                prev_img = previous_layer_image.convert("RGBA")
                prev_datas = prev_img.getdata()
                new_prev_data = []
                for item in prev_datas:
                    # Check if pixel is a brick (not white background or light-grey grid).
                    if not (item[0] > 210 and item[1] > 210 and item[2] > 210):
                        new_prev_data.append((200, 200, 200, 100)) # Light semi-transparent grey
                    else:
                        new_prev_data.append(item) # Keep background and grid lines as they are.
                prev_img.putdata(new_prev_data)

                # --- Process current layer ---
                # Make the white background of the current layer transparent.
                current_img = current_img_with_grid.convert("RGBA")
                current_datas = current_img.getdata()
                new_current_data = []
                for item in current_datas:
                    if item[0] > 250 and item[1] > 250 and item[2] > 250: # is white
                        new_current_data.append((255, 255, 255, 0)) # Make transparent
                    else:
                        new_current_data.append(item)
                current_img.putdata(new_current_data)
                
                # Composite the current layer over the modified previous layer.
                composite = Image.alpha_composite(prev_img, current_img)
                composite.save(current_img_path, "PNG")
            
            # Keep a clean copy of the current image to be used as the 'previous' in the next iteration.
            previous_layer_image = current_img_with_grid.copy()


def brick_model_to_array(brick_model):
    """Converts a BrickModel object back to a 3D NumPy array."""
    size_x, size_y, size_z = brick_model.size
    array = np.zeros((size_z, size_y, size_x), dtype=int)
    if brick_model.layers:
        for layer_num, layer_bricks in brick_model.layers.items():
            for brick in layer_bricks:
                x, y = brick['position']
                width, height = brick['size']
                if layer_num < size_z:
                    array[layer_num, y:y + height, x:x + width] = 1
    return array


def remove_hanging_bricks(brick_model):
    """
    Removes bricks that are not supported by the layer below by operating on the model's array representation.
    """
    
    array = brick_model_to_array(brick_model)
    
    # Iterate from the second layer upwards
    for z in range(1, array.shape[0]):
        current_layer = array[z, :, :]
        lower_layer = array[z - 1, :, :]
        
        # Find bricks (connected components) in the current layer
        labeled_layer, num_features = numpy_label(current_layer)
        
        if num_features > 0:
            for i in range(1, num_features + 1):
                brick_mask = (labeled_layer == i)
                
                # Check if this brick is supported by the layer below
                # Support is defined as any overlap between the brick's footprint and the lower layer
                if not np.any(brick_mask & (lower_layer > 0)):
                    array[z, :, :][brick_mask] = 0 # Remove unsupported brick
    
    # After removing hanging bricks, we need a new brick_model.
    # We can reuse the Brickalizer to convert the array back to a model.
    # Note: This requires the brick_set which is available in the calling context.
    # This function will now return the array, and the conversion will happen in the route.
    return array


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('how_to_use.html')

@app.route('/convert')
def convert_page():
    existing_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.stl') and not f.endswith('_brick_model.stl')]
    existing_files.sort(key=lambda filename: (filename not in STARTER_MODEL_LABELS, filename.lower()))
    return render_template(
        'convert.html',
        existing_files=existing_files,
        starter_model_labels=STARTER_MODEL_LABELS,
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    existing_file = request.form.get('existing_file', '')
    
    if existing_file:
        filename = existing_file
        stl_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(stl_file_path):
            return redirect(url_for('index'))
    else:
        if 'file' not in request.files:
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))
        if not (file and allowed_file(file.filename)):
            return redirect(url_for('index'))
        filename = secure_filename(file.filename)
        stl_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(stl_file_path)
    
    if True:

        grid_voxel_count = int(request.form['grid_voxel_count'])
        grid_direction = request.form['grid_direction']
        extract_shell = 'extract_shell' in request.form
        overlay_layers = 'overlay_layers' in request.form
        generate_pdf = 'generate_pdf' in request.form
        color_by_shape = 'color_by_shape' in request.form
        remove_hanging = 'remove_hanging_bricks' in request.form
        
        # Clean up previous results for this file to avoid conflicts.
        filename_base = os.path.splitext(filename)[0]
        image_folder_path = os.path.join(app.config['RESULT_FOLDER'], filename_base)
        model_path = os.path.join(app.config['RESULT_FOLDER'], f"{filename_base}_brick_model.stl")
        pdf_path = os.path.join(app.config['RESULT_FOLDER'], f"{filename_base}_instructions.pdf")

        if os.path.exists(image_folder_path):
            shutil.rmtree(image_folder_path)
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # --- Brickalize logic ---
        brick_set = BrickSet([Brick(1, 2), Brick(1, 4), Brick(2, 2), Brick(1, 1), Brick(1, 3), Brick(2, 4), Brick(1, 6), Brick(1, 1, True), Brick(1, 2, True)])
        brick_array = Brickalizer.voxelize_stl(stl_file_path, grid_voxel_count, grid_direction, fast_mode=True)
        
        if extract_shell:
            processing_array = Brickalizer.extract_shell_from_3d_array(brick_array)
        else:
            processing_array = brick_array

        brick_model = Brickalizer.array_to_brick_model(processing_array, brick_set)
        
        if not extract_shell and remove_hanging:
            modified_array = remove_hanging_bricks(brick_model)
            brick_model = Brickalizer.array_to_brick_model(modified_array, brick_set)

        brick_model.normalize()

        # --- Color and Image Generation ---
        # Get brick data for coloring and PDF legend
        brick_shapes = set()
        if brick_model.layers:
            for layer_num, layer_bricks in brick_model.layers.items():
                for brick in layer_bricks:
                    brick_shapes.add(tuple(sorted(brick['size'])))
        shape_colors = {shape: (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)) for shape in brick_shapes}
        
        # Save the model as mesh before generating images
        filename_base = os.path.splitext(filename)[0]
        model_filename = f"{filename_base}_brick_model.stl"
        model_path = os.path.join(app.config['RESULT_FOLDER'], model_filename)
        mesh_list = BrickModelVisualizer.draw_model_individual_bricks(brick_model)
        BrickModelVisualizer.save_model(mesh_list, file_path=model_path)

        # Save the model as images
        image_folder = os.path.join(app.config['RESULT_FOLDER'], filename_base)
        os.makedirs(image_folder, exist_ok=True)
        
        num_cells_x = brick_model.size[0]
        num_cells_y = brick_model.size[1]

        if color_by_shape:
            generate_custom_layer_images(brick_model, image_folder, shape_colors)
        else:
            BrickModelVisualizer.save_as_images(brick_model, dir_path=image_folder)
            # Draw grid and outlines on the default images
            default_image_files = sorted(os.listdir(image_folder), key=lambda f: int(re.search(r'(\d+)', f).group(1)))
            for image_file in default_image_files:
                layer_num = int(re.search(r'(\d+)', image_file).group(1))
                image_path = os.path.join(image_folder, image_file)
                draw_grid_on_image(image_path, num_cells_x, num_cells_y)
                draw_brick_outlines_on_image(image_path, layer_num, brick_model)
        
        # Get the list of generated files for further processing and rendering.
        image_files = sorted(os.listdir(image_folder), key=lambda f: int(re.search(r'(\d+)', f).group(1)))
        
        if overlay_layers:
            overlay_image_layers(image_files, image_folder)

        # Generate PDF if requested
        if generate_pdf:
            pdf_filename = f"{filename_base}_instructions.pdf"
            create_pdf_instructions(image_files, image_folder, pdf_path, brick_model, shape_colors)
        else:
            pdf_filename = None

        # Upload files to object storage for persistent access
        stl_object_name = None
        pdf_object_name = None
        
        if os.path.exists(model_path):
            stl_object_name = upload_to_object_storage(model_path, f"models/{model_filename}")
        
        if generate_pdf and os.path.exists(pdf_path):
            pdf_object_name = upload_to_object_storage(pdf_path, f"pdfs/{pdf_filename}")

        return render_template('results.html', 
                             image_files=image_files, 
                             model_filename=model_filename, 
                             result_folder=os.path.splitext(filename)[0], 
                             cache_buster=int(time.time()), 
                             pdf_filename=pdf_filename,
                             stl_object_name=stl_object_name,
                             pdf_object_name=pdf_object_name)

@app.route('/history')
def history():
    results = [d for d in os.listdir(app.config['RESULT_FOLDER']) if os.path.isdir(os.path.join(app.config['RESULT_FOLDER'], d))]
    return render_template('history.html', results=results)

@app.route('/history/<result_folder>')
def view_result(result_folder):
    image_folder = os.path.join(app.config['RESULT_FOLDER'], result_folder)
    image_files = sorted(os.listdir(image_folder), key=lambda f: int(re.search(r'(\d+)', f).group(1)))
    model_filename = f"{result_folder}_brick_model.stl"
    pdf_filename = f"{result_folder}_instructions.pdf"
    
    if not os.path.exists(os.path.join(app.config['RESULT_FOLDER'], pdf_filename)):
        pdf_filename = None

    return render_template('results.html', image_files=image_files, model_filename=model_filename, result_folder=result_folder, cache_buster=int(time.time()), pdf_filename=pdf_filename)

@app.route('/results/<result_folder>/<filename>')
def result_image(result_folder, filename):
    return send_from_directory(os.path.join(app.config['RESULT_FOLDER'], result_folder), filename)

@app.route('/download/<filename>')
def download_model(filename):
    # Try object storage first
    object_name = f"models/{filename}"
    download_url = get_object_download_url(object_name)
    if download_url:
        return redirect(download_url)
    # Fall back to local file with explicit mimetype
    local_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(local_path):
        try:
            return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True, mimetype='application/sla')
        except OSError:
            return "File not available", 404
    return "File not found", 404

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    # Try object storage first
    object_name = f"pdfs/{filename}"
    download_url = get_object_download_url(object_name)
    if download_url:
        return redirect(download_url)
    # Fall back to local file
    local_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(local_path):
        try:
            return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True, mimetype='application/pdf')
        except OSError:
            return "File not available", 404
    return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
