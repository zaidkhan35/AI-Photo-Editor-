# 🖼️ AI Photo Editor

> AI Photo Editor built with Python & OpenCV across 6 phases covering basic editing, artistic effects like cartoon and sketch, face detection, AI background removal, auto enhancement, super resolution, text overlay, and a FastAPI REST API with a dark-themed HTML web interface.

---

## 📌 Project Overview

This is a full computer vision project built with Python and OpenCV across 6 progressive phases, designed as a complete learning roadmap from basic image processing to AI-powered features and a production-ready web application.

---

## 🗂️ Project Structure

```
ai_photo_editor/
│
├── backend/
│   ├── main.py                  ← FastAPI REST API
│   ├── basic_editor.py          ← Phase 1 pipeline
│   ├── artistic_effects.py      ← Phase 2 pipeline
│   ├── face_detection.py        ← Phase 3 pipeline
│   ├── ai_features.py           ← Phase 4 pipeline
│   ├── advanced_features.py     ← Phase 5 pipeline
│   └── requirements.txt
│
└── frontend/
    ├── index.html               ← Main web page
    ├── style.css                ← Dark theme styling
    └── app.js                   ← API calls and UI logic
```

---

## 🚀 Phases

### Phase 1 — Basic Editor
Core image editing operations built with OpenCV.

| Feature | Description |
|---|---|
| Brightness | Adjust image lightness using HSV color space |
| Contrast | Multiply pixel values to spread or flatten tones |
| Blur | Gaussian blur with adjustable kernel size |
| Sharpen | Convolution kernel to enhance edges |
| Crop | Cut a region of interest from the image |
| Resize | Scale image up or down by percentage |
| Rotate | Rotate image by any angle 0–360 |
| Flip | Mirror horizontally, vertically, or both |
| Grayscale | Convert color image to black and white |

**Concepts learned:** image loading, pixel operations, color spaces, convolution kernels, OpenCV basics

---

### Phase 2 — Artistic Effects
Creative filters applied on top of the Phase 1 pipeline.

| Effect | Description |
|---|---|
| Cartoon | Bilateral filter + edge overlay |
| Sketch B&W | Pencil drawing effect using divide blend |
| Sketch Color | Color pencil drawing effect |
| Stylization | OpenCV painterly stylization filter |
| Edge Detection | Canny edge detection inverted |
| Emboss | 3D raised texture effect |
| Sepia | Warm vintage tone using color matrix |

**Concepts learned:** edge detection, Canny algorithm, bilateral filtering, color transformations, blending modes

---

### Phase 3 — Face Detection
Detect and process faces in images using Haar Cascades.

| Mode | Description |
|---|---|
| Bounding Boxes | Draw green rectangles around faces |
| Blur Faces | Apply Gaussian blur to each face |
| Pixelate Faces | Mosaic/pixelate effect on faces |
| Highlight Faces | Darken background, spotlight faces |

**Concepts learned:** Haar Cascades, face detection, region of interest (ROI), bounding boxes

---

### Phase 4 — AI Features
AI-powered image processing features.

| Feature | Description |
|---|---|
| Background Removal | Remove image background using rembg |
| Denoise | Remove grain and noise using NLMeans |
| Auto Enhance | Smart contrast boost using CLAHE |
| Super Resolution | Upscale image quality using DNN |

**Concepts learned:** deep learning integration, segmentation, CLAHE, super resolution networks

---

### Phase 5 — Advanced Features
Professional-grade editing tools.

| Feature | Description |
|---|---|
| Text Overlay | Add custom text with font, size, color, position |
| Quality Enhancement | Unsharp masking + CLAHE sharpening |
| Image Compression | Control output file size by JPEG quality |

**Concepts learned:** PIL font rendering, unsharp masking, image compression

---

### Phase 6 — Web Application
Full REST API backend and professional frontend.

| Part | Technology | Description |
|---|---|---|
| Backend | FastAPI | REST API receives image + settings, runs pipeline, returns result |
| Frontend | HTML + CSS + JS | Dark themed UI with sliders, tabs, split preview, download |

**Concepts learned:** REST API design, multipart form data, base64 encoding, fetch API, async JavaScript

---

## 🛠️ Installation

### Requirements
```bash
pip install -r backend/requirements.txt
```

### Dependencies
```
fastapi
uvicorn
opencv-python
numpy
pillow
python-multipart
```

### Optional AI Features
```bash
pip install "rembg[cpu]"    # background removal
```

---

## ▶️ How to Run

### Step 1 — Start the backend
```bash
cd backend
uvicorn main:app --reload
```
Backend runs at: `http://localhost:8000`

### Step 2 — Open the frontend
Open `frontend/index.html` directly in your browser. No extra server needed.

### Step 3 — Use the editor
- Upload any image by clicking or drag and drop
- Use the 5 phase tabs to access all features
- Sliders auto-apply after 600ms
- Click Apply for instant processing
- Switch between Split / Original / Result views
- Click Download to save your edited image

---

## 🔌 API Reference

### Health Check
```
GET http://localhost:8000/
```
```json
{ "status": "AI Photo Editor API is running" }
```

### Process Image
```
POST http://localhost:8000/process
Content-Type: multipart/form-data
```

**Parameters:**
```
image         file     image file (required)

# Phase 1
brightness    int      0–200    default 100
contrast      int      1–40     default 10
blur          int      1–51     default 1
sharpen       int      0/1      default 0
grayscale     int      0/1      default 0
scale         int      10–300   default 100
rotate        int      0–360    default 0
flip          int      0–3      default 0
crop_x        int               default 0
crop_y        int               default 0
crop_w        int               default 0
crop_h        int               default 0

# Phase 2
cartoon       int      0–100    default 0
sketch_bw     int      0–100    default 0
sketch_color  int      0–100    default 0
stylize       int      0–100    default 0
edges         int      0–100    default 0
emboss        int      0–100    default 0
sepia         int      0–100    default 0

# Phase 3
face_mode     int      0–4      default 0
min_neighbors int      1–20     default 5
scale_factor  int      11–20    default 12

# Phase 4
bg_remove     int      0/1      default 0
denoise       int      0–100    default 0
auto_enhance  int      0–100    default 0
super_res     int      0–100    default 0

# Phase 5
text_on       int      0/1      default 0
text_content  str               default ""
text_x        int      0–100    default 10
text_y        int      0–100    default 10
text_size     int      10–200   default 40
text_r        int      0–255    default 255
text_g        int      0–255    default 255
text_b        int      0–255    default 255
enhance_qual  int      0–100    default 0
```

**Response:**
```json
{
  "result":     "data:image/jpeg;base64,...",
  "face_count": 2,
  "width":      1280,
  "height":     720
}
```

---

## 📸 Features at a Glance

```
✅ Brightness / Contrast / Blur / Sharpen
✅ Crop / Resize / Rotate / Flip
✅ Grayscale
✅ Cartoon Effect
✅ Pencil Sketch B&W and Color
✅ Stylization / Edge Detection / Emboss / Sepia
✅ Face Detection and Counting
✅ Face Blur / Pixelate / Highlight
✅ AI Background Removal
✅ Noise Reduction
✅ One-Click Auto Enhancement
✅ Super Resolution Upscaling
✅ Custom Text Overlay
✅ Quality Enhancement
✅ Image Compression
✅ REST API Backend
✅ Dark Themed Web Interface
✅ Split View Comparison
✅ One-Click Download
```

---

## 🧠 Concepts Covered

- OpenCV image processing
- Color spaces (BGR, HSV, LAB)
- Convolution kernels
- Gaussian blur and Canny edge detection
- Haar Cascade face detection
- Region of interest (ROI)
- CLAHE contrast enhancement
- Unsharp masking
- Deep learning background removal
- FastAPI REST API design
- HTML / CSS / JavaScript frontend
- Base64 image encoding
- Async fetch API

---

## 👤 Author

Built as a complete computer vision learning roadmap — from OpenCV basics to a production-ready AI web application.

---

## 📄 License

MIT License — free to use, modify, and share.
