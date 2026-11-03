"""
AI-Based Vessel Segmentation and Rim Detection System
=====================================================

Main Flask application entry point.

- Loads the YOLO segmentation model **once** at startup.
- Exposes routes for uploading images and processing them.
- Returns JSON results so the frontend can update progressively.

Architecture Notes:
    All heavy lifting is delegated to modules under ``utils/``.
    The app itself is a thin orchestration layer, making it easy to
    add future stages (e.g. DINOv2 feature extraction, vector
    similarity search) without touching existing code.
"""

import os
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

try:
    yolo_model = load_model(MODEL_PATH)
    logger.info("✓ YOLO model ready.")
except FileNotFoundError:
    logger.warning(
        "⚠  Model file not found at '%s'. "
        "The app will start, but /process will return errors until "
        "you place best.pt in the models/ directory.",
        MODEL_PATH,
    )
except RuntimeError as e:
    logger.error("✗ Failed to load model: %s", e)


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
    Accept an uploaded image, run segmentation + rim extraction,
    and return paths + timings as JSON.

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
    upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(upload_path)
    logger.info("Saved upload → %s", upload_path)

    try:
        # ── 1. Segmentation ──────────────────────────────
        seg_result = run_segmentation(yolo_model, upload_path)

        seg_filename = f"seg_{safe_name}"
        seg_path = os.path.join(OUTPUT_FOLDER, seg_filename)
        cv2.imwrite(seg_path, seg_result["segmented_image"])

        # ── 2. Rim extraction ────────────────────────────
        original_image = cv2.imread(upload_path)
        rim_result = extract_rim(
            mask=seg_result["mask"],
            original_image=original_image,
        )

        rim_filename = f"rim_{safe_name}"
        rim_path = os.path.join(OUTPUT_FOLDER, rim_filename)
        cv2.imwrite(rim_path, rim_result["rim_image"])

        # Also save the rim-on-black version
        rim_black_filename = f"rim_black_{safe_name}"
        rim_black_path = os.path.join(OUTPUT_FOLDER, rim_black_filename)
        cv2.imwrite(rim_black_path, rim_result["rim_on_black"])

        # ── 3. Compute totals ────────────────────────────
        total_time = seg_result["time"] + rim_result["time"]

        return jsonify({
            "success": True,
            "original_url": url_for("static", filename=f"uploads/{safe_name}"),
            "segmented_url": url_for("static", filename=f"outputs/{seg_filename}"),
            "rim_url": url_for("static", filename=f"outputs/{rim_filename}"),
            "rim_black_url": url_for("static", filename=f"outputs/{rim_black_filename}"),
            "has_detections": seg_result["has_detections"],
            "has_rim": rim_result["has_rim"],
            "confidence": seg_result["confidence"],
            "timings": {
                "segmentation": round(seg_result["time"], 4),
                "rim_extraction": round(rim_result["time"], 4),
                "total": round(total_time, 4),
            },
        })

    except ValueError as e:
        logger.exception("Validation error during processing.")
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.exception("Unexpected error during processing.")
        return jsonify({"success": False, "error": f"Processing failed: {e}"}), 500


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
