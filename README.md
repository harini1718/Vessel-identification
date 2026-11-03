# 🔬 AI-Based Vessel Segmentation and Rim Detection System

A production-quality Flask web application that segments vessels from top-view images using a locally trained **YOLOv11 segmentation model** and extracts the **outer rim** using a dedicated **OpenCV** algorithm. All processing runs **100% locally** — no cloud APIs, no Roboflow, no external inference services.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **YOLO Segmentation** | Runs a custom-trained YOLOv11 instance segmentation model locally |
| **OpenCV Rim Extraction** | Separate pure-OpenCV pipeline to compute the vessel's outer rim |
| **Multiple Image Upload** | Drag-and-drop or browse; each image is processed independently |
| **Progressive UI** | Real-time pipeline status: Upload → Segmentation → Rim → Complete |
| **Performance Timings** | Per-image breakdown of segmentation, rim extraction, and total time |
| **Error Handling** | Graceful handling of invalid images, missing model, and empty predictions |
| **Extensible Architecture** | Modular design ready for DINOv2 / vector-search extensions |

---

## 📁 Project Structure

```
vessel-segmentation-app/
│
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── models/
│   └── best.pt                 # Your trained YOLO segmentation weights
│
├── templates/
│   └── index.html              # Main web interface
│
├── static/
│   ├── css/
│   │   └── style.css           # Application stylesheet
│   ├── uploads/                # Uploaded images (auto-created)
│   └── outputs/                # Processed output images (auto-created)
│
└── utils/
    ├── __init__.py
    ├── segmentation.py         # YOLO model loading & inference
    ├── rim_extraction.py       # OpenCV rim extraction algorithm
    └── timer.py                # Context-manager timer utility
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (comes with Python)
- A trained YOLO segmentation model (`best.pt`)

### Steps

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd vessel-segmentation-app
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place your model weights**

   Copy your trained `best.pt` file into the `models/` directory:

   ```
   models/best.pt
   ```

---

## ▶️ How to Run

```bash
python app.py
```

The application starts at **http://localhost:5000**.

Open the URL in your browser and start uploading vessel images.

### Production Mode (optional)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 🧪 How to Test

1. Open **http://localhost:5000** in your browser.
2. Click **Select Images** or drag-and-drop one or more top-view vessel images.
3. Watch the pipeline progress through each stage:
   - **Upload** → **Segmentation** → **Rim Extraction** → **Completed**
4. Each result card shows:
   - **Original Image** — the unmodified upload
   - **Segmented Vessel** — YOLO mask overlay (green) with contour outlines
   - **Outer Rim** — the extracted vessel rim drawn in red on the original
5. Timing breakdown appears below each image set.

### Health Check

```bash
curl http://localhost:5000/health
```

Returns `{"model_loaded": true, "status": "ok"}` when everything is ready.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| Ultralytics | YOLOv11 model loading & inference |
| OpenCV (cv2) | Image processing & rim extraction |
| NumPy | Array operations |
| PyTorch | Deep learning backend for YOLO |
| Pillow | Image format support |

---

## 🔧 Rim Extraction Algorithm

The rim is **not** taken from the YOLO output. A dedicated OpenCV pipeline in `utils/rim_extraction.py` computes it:

```
Segmentation Mask
       ↓
  Binarise (threshold)
       ↓
  Keep largest connected component
       ↓
  Morphological closing (fill gaps)
       ↓
  Find external contour (cv2.RETR_EXTERNAL)
       ↓
  Smooth contour (cv2.approxPolyDP)
       ↓
  Draw outer contour → Rim Image
```

Parameters (thickness, smoothing, kernel size) are configurable via function arguments.

**Performance:** Targets < 50 ms per image on CPU.

---

## 🛣️ Future Work

The architecture is designed for easy extension:

```
Current Pipeline                Future Extension
─────────────────              ─────────────────
Upload Image                   Upload Image
     ↓                              ↓
YOLO Segmentation              YOLO Segmentation
     ↓                              ↓
Rim Extraction                 Rim Extraction
     ↓                              ↓
Display Results                DINOv2 Feature Extraction
                                    ↓
                               Vector Embedding
                                    ↓
                               Vector Similarity Search
                                    ↓
                               Registered Vessel Identification
```

To add these stages, create new modules under `utils/` (e.g., `utils/feature_extraction.py`, `utils/similarity.py`) and register new processing steps in `app.py` without modifying existing code.

---

## 📸 Screenshots

> *Add screenshots of the running application here.*

| Upload Page | Segmented Vessel | Extracted Rim |
|:-----------:|:----------------:|:-------------:|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 📄 License

This project is for internal/research use. Add your preferred license here.
