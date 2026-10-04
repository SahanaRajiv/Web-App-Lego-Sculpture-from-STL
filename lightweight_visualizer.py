"""
Lightweight replacement for BrickModelVisualizer using trimesh instead of open3d.
This eliminates the 1.1GB open3d dependency.
"""

import trimesh
import numpy as np
import cv2
from tqdm import tqdm
import math

class LightweightVisualizer:
    @staticmethod
    def draw_brick(position: tuple, size: tuple):
        """Create a box mesh using trimesh."""
        l, w, h = size
        x, y, z = position
        box = trimesh.creation.box(extents=[w, l, h])
        box.apply_translation([x + w/2, y + l/2, z + h/2])
        return box
    
    @classmethod
    def draw_model_individual_bricks(cls, brick_model) -> list:
        """Draw multiple bricks and return mesh list."""
        mesh_list = []
        for z in sorted(brick_model.layers.keys()):
            for brick in brick_model.layers[z]:
                brick_size = (brick["size"][1], brick["size"][0], brick_model.layer_height)
                brick_position = (brick["position"][0], brick["position"][1], z * brick_model.layer_height)
                box = cls.draw_brick(brick_position, brick_size)
                mesh_list.append({'name': str(brick_position), 'geometry': box, 'support': brick["support"]})
        return mesh_list

    @staticmethod
    def save_model(mesh_list: list, file_path: str) -> bool:
        """Save the model to an STL file using trimesh."""
        try:
            meshes = [item['geometry'] for item in mesh_list]
            combined = trimesh.util.concatenate(meshes)
            combined.export(file_path, file_type='stl')
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def save_as_images(
        cls,
        brick_model,
        dir_path: str,
        brick_color: tuple = (0, 255, 255),
        support_color: tuple = (200, 200, 255),
        add_lego_overlay: bool = True,
        show_ghost_layer: bool = False,
        pixels_per_stud: int = 20,
        line_thickness: float = 0.05
    ):
        """Convert the model to images using OpenCV."""
        empty = np.full(([brick_model.size[1] * pixels_per_stud, brick_model.size[0] * pixels_per_stud, 3]), 255, dtype=np.uint8)
        if show_ghost_layer: 
            ghost_layer = np.copy(empty)
        if add_lego_overlay: 
            shadow_img = cls._generate_lego_shadow(pixels_per_stud)

        for z, layer in tqdm(sorted(brick_model.layers.items()), desc="Creating images..."):
            if show_ghost_layer:
                image = np.copy(ghost_layer)
                ghost_layer = np.copy(empty)
            else: 
                image = np.copy(empty)
            for b in layer:
                w, l = b["size"]
                x = b["position"][0] - brick_model.min[0]
                y = brick_model.size[1] - b["position"][1] - l
                
                if b["support"]:
                    color = support_color
                    outline_color = tuple(int(c*0.8) for c in support_color)
                    if show_ghost_layer:
                        transparent_color = tuple(int(c+(255-c)//2) for c in color)
                else:
                    color = brick_color
                    outline_color = tuple(int(c*0.8) for c in brick_color)
                    if show_ghost_layer:
                        transparent_color = tuple(int(c+(255-c)//1.5) for c in color)

                cv2.rectangle(image, (x * pixels_per_stud, y * pixels_per_stud), ((x + w) * pixels_per_stud, (y + l) * pixels_per_stud), color, -1)
                if show_ghost_layer: 
                    cv2.rectangle(ghost_layer, (x * pixels_per_stud, y * pixels_per_stud), ((x + w) * pixels_per_stud, (y + l) * pixels_per_stud), transparent_color, -1)
            
                if pixels_per_stud > 3:
                    cv2.rectangle(image, (x * pixels_per_stud, y * pixels_per_stud), ((x + w) * pixels_per_stud, (y + l) * pixels_per_stud), outline_color, math.ceil(line_thickness * pixels_per_stud))
                
                if pixels_per_stud > 9 and add_lego_overlay:
                    for i in range(w):
                        for j in range(l):
                            image = cls._apply_shadow_with_multiply(image, shadow_img, (x+i) * pixels_per_stud, (y+j) * pixels_per_stud)
            cv2.imwrite(f"{dir_path}/layer_{z}.png", image)

    @staticmethod
    def _generate_lego_shadow(size) -> np.ndarray:
        """Creates a square image with a half-circle shadow effect."""
        scaler = 0.6
        shadow_width = int(size * scaler)
        shadow_thickness = max(1, shadow_width // 10)
        small_shadow_thickness = max(1, shadow_width // 14)
        smaller_shadow_thickness = max(1, shadow_width // 16)
        img = np.ones((size, size), dtype=np.uint8) * 255

        center = (size // 2, size // 2)
        radius = shadow_width // 2

        cv2.ellipse(img, center, (radius, radius), 0, 0, 180, 220, smaller_shadow_thickness)
        cv2.ellipse(img, center, (radius, radius), 0, 20, 160, 180, smaller_shadow_thickness)
        cv2.ellipse(img, center, (radius, radius), 0, 40, 140, 150, small_shadow_thickness)
        cv2.ellipse(img, center, (radius, radius), 0, 60, 120, 150, shadow_thickness)
        cv2.ellipse(img, center, (radius, radius), 0, 80, 100, 127, shadow_thickness)

        if size > 16:
            blur_size = max(3, (shadow_width // 5) | 1)
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        shadow_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        shadow_float = shadow_rgba.astype(np.float32) / 255.0

        return shadow_float

    @staticmethod
    def _apply_shadow_with_multiply(image, shadow, x, y) -> np.ndarray:
        """Apply a shadow to a BGR image at the given position."""
        image_float = image.astype(np.float32) / 255.0
        overlay = np.ones_like(image, dtype=np.float32)
        h, w = shadow.shape[:2]
        try:
            overlay[y:y+h, x:x+w] = shadow
        except:
            pass
        blended = image_float * overlay
        blended = (blended * 255).astype(np.uint8)
        return blended
