"""
AI-Based Vessel Segmentation and Rim Detection System
=====================================================

Main Flask application entry point.

- Loads the YOLO segmentation model **once** at startup.
- `/process` runs segmentation and returns the segmentation result
  immediately (so the UI can display it right after prediction).
- `/extract-rim` runs the OpenCV outer-rim pipeline on the mask saved
  by `/process`.
- Returns JSON results so the frontend can update progressively.

Architecture Notes:
    All heavy lifting is delegated to modules under ``utils/``.
    The app itself is a thin orchestration layer, making it easy to
    add future stages (e.g. DINOv2 feature extraction, vector
    similarity search) without touching existing code.
"""

import os
import time
import uuid
import logging

import cv2
from flask import Flask, render_template, request, jsonify, url_for

from utils.segmentation import load_model, run_segmentation
from utils.rim_extraction import extract_rim

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB per request

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Flask app factory
# ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ──────────────────────────────────────────────────────────
# Load model ONCE at startup
# ──────────────────────────────────────────────────────────
yolo_model = None
_model_load_start = time.perf_counter()

try:
    yolo_model = load_model(MODEL_PATH)
    logger.info(
        "✓ YOLO model ready in %.2f s.",
        time.perf_counter() - _model_load_start,
    )
except FileNotFoundError:
    logger.warning(
        "⚠  Model file not found at '%s' after %.2f s. "
        "The app will start, but /process will return errors until "
        "you place best.pt in the models/ directory.",
        MODEL_PATH,
        time.perf_counter() - _model_load_start,
    )
except RuntimeError as e:
    logger.error(
        "✗ Failed to load model after %.2f s: %s",
        time.perf_counter() - _model_load_start,
        e,
    )


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def _allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _unique_filename(original: str) -> str:
    """Generate a collision-free filename preserving the original extension."""
    ext = original.rsplit(".", 1)[1].lower() if "." in original else "png"
    return f"{uuid.uuid4().hex}.{ext}"


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main upload page."""
    model_loaded = yolo_model is not None
    return render_template("index.html", model_loaded=model_loaded)


@app.route("/process", methods=["POST"])
def process_image():
    """
    Accept an uploaded image, run YOLO segmentation, and return the
    segmentation result immediately.

    The saved mask filename is also returned so the frontend can call
    `/extract-rim` as the next step.

    Expected form field: ``file`` (multipart file upload).
    """
    # ── Guard: model available? ────────────────────────────
    if yolo_model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please place best.pt in the models/ directory and restart the server.",
        }), 503

    # ── Guard: file present? ──────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }), 400

    # ── Save uploaded file ────────────────────────────────
    safe_name = _unique_filename(file.filename)
    stem = os.path.splitext(safe_name)[0]
    upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(upload_path)
    logger.info("Saved upload → %s", upload_path)

    try:
        # ── 1. Segmentation (shown immediately in the UI) ──
        seg_result = run_segmentation(yolo_model, upload_path)
        logger.info(
            "Segmentation finished in %.4f s (has_detections=%s).",
            seg_result["time"],
            seg_result["has_detections"],
        )

        seg_filename = f"seg_{stem}.png"
        seg_path = os.path.join(OUTPUT_FOLDER, seg_filename)
        cv2.imwrite(seg_path, seg_result["segmented_image"])

        # Persist the raw mask so /extract-rim can reuse it without
        # running the model a second time.
        mask_filename = f"mask_{stem}.png"
        mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
        cv2.imwrite(mask_path, seg_result["mask"])

        return jsonify({
            "success": True,
            "original_url": url_for("static", filename=f"uploads/{safe_name}"),
            "segmented_url": url_for("static", filename=f"outputs/{seg_filename}"),
            "mask_filename": mask_filename,
            "original_filename": safe_name,
            "has_detections": seg_result["has_detections"],
            "confidence": seg_result["confidence"],
            "timings": {
                "segmentation": round(seg_result["time"], 4),
            },
        })

    except ValueError as e:
        logger.exception("Validation error during segmentation.")
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.exception("Unexpected error during segmentation.")
        return jsonify({"success": False, "error": f"Processing failed: {e}"}), 500


@app.route("/extract-rim", methods=["POST"])
def extract_rim_image():
    """
    Run the OpenCV outer-rim pipeline on a mask produced by `/process`.

    Expected JSON body:
        {
            "mask_filename": "mask_<id>.png",
            "original_filename": "<id>.png"
        }
    """
    data = request.get_json(silent=True) or {}

    mask_filename = os.path.basename(str(data.get("mask_filename", "")))
    original_filename = os.path.basename(str(data.get("original_filename", "")))

    if not mask_filename or not original_filename:
        return jsonify({
            "success": False,
            "error": "Missing mask or original image reference.",
        }), 400

    mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
    original_path = os.path.join(UPLOAD_FOLDER, original_filename)

    if not os.path.isfile(mask_path) or not os.path.isfile(original_path):
        return jsonify({
            "success": False,
            "error": "Source images not found. Please upload the image again.",
        }), 400

    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(original_path)

        if mask is None or original_image is None:
            raise ValueError("Could not read segmentation mask or original image.")

        # ── 2. Outer rim extraction ────────────────────────
        rim_result = extract_rim(
            mask=mask,
            original_image=original_image,
        )

        stem = os.path.splitext(original_filename)[0]

        # Actual rim image: real pixels on a white background (reference style).
        rim_filename = f"rim_{stem}.png"
        rim_path = os.path.join(OUTPUT_FOLDER, rim_filename)
        cv2.imwrite(rim_path, rim_result["rim_photo_white"])

        # Actual rim image with a transparent background.
        rim_transparent_filename = f"rim_transparent_{stem}.png"
        rim_transparent_path = os.path.join(OUTPUT_FOLDER, rim_transparent_filename)
        cv2.imwrite(rim_transparent_path, rim_result["rim_photo_transparent"])

        # Black-and-white binary rim mask (single channel).
        rim_mask_filename = f"rim_mask_{stem}.png"
        rim_mask_path = os.path.join(OUTPUT_FOLDER, rim_mask_filename)
        cv2.imwrite(rim_mask_path, rim_result["rim_mask"])

        return jsonify({
            "success": True,
            "rim_url": url_for("static", filename=f"outputs/{rim_filename}"),
            "rim_transparent_url": url_for("static", filename=f"outputs/{rim_transparent_filename}"),
            "rim_mask_url": url_for("static", filename=f"outputs/{rim_mask_filename}"),
            "has_rim": rim_result["has_rim"],
            "timings": {
                "rim_extraction": round(rim_result["time"], 4),
            },
        })

    except ValueError as e:
        logger.exception("Validation error during rim extraction.")
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.exception("Unexpected error during rim extraction.")
        return jsonify({"success": False, "error": f"Rim extraction failed: {e}"}), 500


@app.route("/health")
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": yolo_model is not None,
    })


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
