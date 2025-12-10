import io
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps, ImageDraw
from colorsys import rgb_to_hsv, hsv_to_rgb
from transformers import AutoProcessor, OneFormerForUniversalSegmentation
import json
import base64
import cv2
import logging
from flask_cors import CORS
import os

# Initialize Flask
app = Flask(__name__)
CORS(app) # Enable CORS for all routes and origins

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model directly
try:
    processor = AutoProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    model.eval() # Set to evaluation mode
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    logger.info(f"Model shi-labs/oneformer_ade20k_swin_large loaded on {DEVICE}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# ADE20K class IDs for indoor elements
SEGMENT_CLASSES = {
    "wall": 0,
    "floor": 2,
    "ceiling": 5,
    "window": 8,
    "cabinet": 9,
    "chair": 12,
    "sofa": 14,
    "door": 15,
    "picture": 18,
    "light": 21,
    "pillow": 33,
    "mirror": 40,
    "armchair": 47,
    "television": 50,
    "lamp": 20
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def blend_color_with_mask(original_image, mask, target_rgb, opacity=0.85):
    original_image = original_image.convert("RGB")
    target_h, target_s, _ = rgb_to_hsv(target_rgb[0]/255.0, target_rgb[1]/255.0, target_rgb[2]/255.0)
    original_array = np.array(original_image)
    mask_array = np.array(mask)
    recolored_array = original_array.copy()
    segment_pixels = mask_array > 0
    if not np.any(segment_pixels):
        logger.warning("No pixels found in mask for recoloring")
        return original_image
    original_segment_pixels = original_array[segment_pixels]
    original_segment_pixels_norm = original_segment_pixels / 255.0
    original_segment_pixels_hsv = np.array([rgb_to_hsv(p[0], p[1], p[2]) for p in original_segment_pixels_norm])
    new_segment_pixels_hsv = np.hstack([
        np.full((len(original_segment_pixels_hsv), 1), target_h),
        np.full((len(original_segment_pixels_hsv), 1), target_s),
        original_segment_pixels_hsv[:, 2:3]
    ])
    new_segment_pixels_rgb_norm = np.array([hsv_to_rgb(p[0], p[1], p[2]) for p in new_segment_pixels_hsv])
    new_segment_pixels_rgb = (new_segment_pixels_rgb_norm * 255).astype(np.uint8)
    blended_segment_pixels = (original_segment_pixels * (1 - opacity) + new_segment_pixels_rgb * opacity).astype(np.uint8)
    recolored_array[segment_pixels] = blended_segment_pixels
    return Image.fromarray(recolored_array)

def blend_texture_with_mask(original_image, mask, texture_image, texture_scale=0.5, texture_opacity=0.85):
    """
    Blends a texture realistically using Multiply blend, with controllable
    scale and opacity for the final effect.
    """
    original_image = original_image.convert("RGB")
    texture_image = texture_image.convert("RGB")
    mask = mask.convert("L")
    # --- SCALING LOGIC (Unchanged) ---
    scaled_width = int(texture_image.width * texture_scale)
    scaled_height = int(texture_image.height * texture_scale)
    if scaled_width < 1 or scaled_height < 1:
        scaled_width, scaled_height = 1, 1
    scaled_texture = texture_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
   
    # --- TILING LOGIC (Unchanged) ---
    tiled_texture = Image.new('RGB', original_image.size)
    for i in range(0, original_image.width, scaled_texture.width):
        for j in range(0, original_image.height, scaled_texture.height):
            tiled_texture.paste(scaled_texture, (i, j))
    # --- ADVANCED BLENDING LOGIC ---
    original_array = np.array(original_image).astype(float)
    tiled_texture_array = np.array(tiled_texture).astype(float)
    # 1. Apply the "Multiply" blend mode to preserve shadows
    multiplied_blend_array = (original_array / 255.0 * tiled_texture_array / 255.0) * 255.0
    # 2. Apply opacity
    # Linearly interpolate between the original image and the textured image
    opacity_blended_array = (original_array * (1 - texture_opacity) + multiplied_blend_array * texture_opacity)
   
    # Ensure values are within the valid 0-255 range and convert to integer
    final_blended_array = np.clip(opacity_blended_array, 0, 255).astype(np.uint8)
   
    blended_image = Image.fromarray(final_blended_array)
   
    # 3. Composite the result using the mask (Unchanged)
    final_image = Image.composite(blended_image, original_image, mask)
    # --- END BLENDING LOGIC ---
   
    return final_image.convert("RGB")

def refine_mask(mask):
    mask = mask.astype(np.uint8)
    # Morphological closing to fill holes
    kernel = np.ones((10, 10), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Morphological opening to remove noise
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Contour filtering to keep large regions
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 1000 # Increased for stricter filtering
    filtered_mask = np.zeros_like(opened_mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return filtered_mask

@app.route("/textures", methods=["GET"])
def get_textures():
    texture_dir = "textures"
    if not os.path.exists(texture_dir):
        return jsonify({"error": "Textures directory not found"}), 500
    textures = []
    base_url = request.host_url
    for filename in os.listdir(texture_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            textures.append({
                "name": os.path.splitext(filename)[0],
                "url": f"{base_url}get_texture/{filename}"
            })
    return jsonify(textures)

@app.route("/get_texture/<filename>", methods=["GET"])
def get_texture(filename):
    texture_dir = "textures"
    return send_file(os.path.join(texture_dir, filename), mimetype='image/png')

@app.route("/segment_areas", methods=["POST"])
def segment_areas():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        input_image = Image.open(file.stream).convert("RGB")
        input_image = ImageOps.exif_transpose(input_image)
        inputs = processor(images=input_image, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_mask = processor.post_process_semantic_segmentation(outputs, target_sizes=[input_image.size[::-1]])[0].cpu().numpy()
        segmented_areas = []
        for label, class_id in SEGMENT_CLASSES.items():
            segment_mask = (pred_mask == class_id).astype(np.uint8) * 255
            if np.any(segment_mask):
                refined_mask = refine_mask(segment_mask)
                if np.any(refined_mask):
                    mask_img = Image.fromarray(refined_mask)
                    buffer = io.BytesIO()
                    mask_img.save(buffer, format="PNG")
                    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    segmented_areas.append({"id": label, "label": label, "mask": mask_base64})
        buffer = io.BytesIO()
        input_image.save(buffer, format="JPEG")
        original_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({
            "original_image": original_image_base64,
            "segmented_areas": segmented_areas
        })
    except Exception as e:
        logger.error(f"Error in segment_areas: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/color_segment", methods=["POST"])
def color_segment():
    data = request.get_json()
    if not data or 'image' not in data or 'mask' not in data or 'color' not in data:
        return jsonify({"error": "Invalid request"}), 400
    try:
        original_image = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert("RGB")
        mask = Image.open(io.BytesIO(base64.b64decode(data['mask']))).convert('L')
        color_hex = data['color']
        target_rgb = hex_to_rgb(color_hex)
        recolored_image = blend_color_with_mask(original_image, mask, target_rgb)
        buffer = io.BytesIO()
        recolored_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return send_file(buffer, mimetype='image/jpeg', as_attachment=False)
    except Exception as e:
        logger.error(f"Error in color_segment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/texture_segment", methods=["POST"])
def texture_segment():
    data = request.get_json()
    if not data or 'image' not in data or 'mask' not in data or 'texture_name' not in data:
        return jsonify({"error": "Invalid request"}), 400
    try:
        original_image = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert("RGB")
        mask = Image.open(io.BytesIO(base64.b64decode(data['mask']))).convert('L')
        texture_name = data['texture_name']
        texture_scale = data.get('scale', 0.5)
        texture_opacity = data.get('opacity', 0.85) # NEW: Get opacity from request
        texture_path = os.path.join("textures", f"{texture_name}.png")
        if not os.path.exists(texture_path):
            texture_path = os.path.join("textures", f"{texture_name}.jpg")
            if not os.path.exists(texture_path):
                return jsonify({"error": "Texture not found"}), 404
        texture_image = Image.open(texture_path)
        textured_image = blend_texture_with_mask(original_image, mask, texture_image, texture_scale, texture_opacity)
        buffer = io.BytesIO()
        textured_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return send_file(buffer, mimetype='image/jpeg', as_attachment=False)
    except Exception as e:
        logger.error(f"Error in texture_segment: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)