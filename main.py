
# FastAPI Backend — REST API

import sys
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import basic_editor
import artistic_effects
import face_detection
import ai_features
import advanced_features as advanced_features_module

app = FastAPI(title="AI Photo Editor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(image, quality=92):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/")
def health():
    return {"status": "AI Photo Editor API is running"}


@app.post("/process")
async def process_image(
    image:             UploadFile = File(...),

    # phase 1
    brightness:        int = Form(100),
    contrast:          int = Form(10),
    blur:              int = Form(1),
    sharpen:           int = Form(0),
    grayscale:         int = Form(0),
    scale:             int = Form(100),
    rotate:            int = Form(0),
    flip:              int = Form(0),
    crop_x:            int = Form(0),
    crop_y:            int = Form(0),
    crop_w:            int = Form(0),
    crop_h:            int = Form(0),

    # phase 2
    cartoon:           int = Form(0),
    sketch_bw:         int = Form(0),
    sketch_color:      int = Form(0),
    stylize:           int = Form(0),
    edges:             int = Form(0),
    emboss:            int = Form(0),
    sepia:             int = Form(0),

    # phase 3
    face_mode:         int = Form(0),
    min_neighbors:     int = Form(5),
    scale_factor_x10:  int = Form(12),

    # phase 4
    bg_remove:         int = Form(0),
    denoise:           int = Form(0),
    auto_enhance:      int = Form(0),
    super_res:         int = Form(0),

    # phase 5
    text_on:           int = Form(0),
    text_content:      str = Form(""),
    text_x:            int = Form(10),
    text_y:            int = Form(10),
    text_size:         int = Form(40),
    text_r:            int = Form(255),
    text_g:            int = Form(255),
    text_b:            int = Form(255),
    enhance_qual:      int = Form(0),
):
    file_bytes = await image.read()
    img        = decode_image(file_bytes)

    if img is None:
        return JSONResponse({"error": "Could not read image"}, status_code=400)

    ih, iw = img.shape[:2]

    pipeline_state = {
        # phase 1
        "brightness":       brightness,
        "contrast":         contrast,
        "blur":             blur,
        "sharpen":          sharpen,
        "grayscale":        grayscale,
        "scale":            scale,
        "rotate":           rotate,
        "flip":             flip,
        "crop_x":           crop_x,
        "crop_y":           crop_y,
        "crop_w":           crop_w if crop_w > 0 else iw,
        "crop_h":           crop_h if crop_h > 0 else ih,
        # phase 2
        "cartoon":          cartoon,
        "sketch_bw":        sketch_bw,
        "sketch_color":     sketch_color,
        "stylize":          stylize,
        "edges":            edges,
        "emboss":           emboss,
        "sepia":            sepia,
        # phase 3
        "face_mode":        face_mode,
        "min_neighbors":    min_neighbors,
        "scale_factor_x10": scale_factor_x10,
        # phase 4
        "bg_remove":        bg_remove,
        "denoise":          denoise,
        "auto_enhance":     auto_enhance,
        "super_res":        super_res,
        # phase 5
        "text_on":          text_on,
        "text_content":     text_content,
        "text_x":           text_x,
        "text_y":           text_y,
        "text_size":        text_size,
        "text_r":           text_r,
        "text_g":           text_g,
        "text_b":           text_b,
        "enhance_qual":     enhance_qual,
    }

    basic_editor.original = img.copy()
    basic_editor.h, basic_editor.w = img.shape[:2]
    basic_editor.state.update(pipeline_state)

    artistic_effects.original = basic_editor.original
    artistic_effects.h = basic_editor.h
    artistic_effects.w = basic_editor.w
    artistic_effects.state = basic_editor.state

    face_detection.original = basic_editor.original
    face_detection.h = basic_editor.h
    face_detection.w = basic_editor.w
    face_detection.state = basic_editor.state

    ai_features.original = basic_editor.original
    ai_features.h = basic_editor.h
    ai_features.w = basic_editor.w
    ai_features.state = basic_editor.state

    advanced_features_module.original = basic_editor.original
    advanced_features_module.h = basic_editor.h
    advanced_features_module.w = basic_editor.w
    advanced_features_module.state = basic_editor.state

    result = advanced_features_module.run_phase5_pipeline()
    face_count = 0

    if hasattr(face_detection, "detect_faces"):
        faces = face_detection.detect_faces(result)
        face_count = len(faces)

    return JSONResponse({
        "result":     encode_image(result),
        "face_count": face_count,
        "width":      result.shape[1],
        "height":     result.shape[0],
    })