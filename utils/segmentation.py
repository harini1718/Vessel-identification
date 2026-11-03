"""
Segmentation Module
===================

Handles all YOLO segmentation logic using a locally trained
Ultralytics YOLOv11 segmentation model.

The model is loaded once at application startup and reused for
every inference request, avoiding costly re-initialization.

Functions:
    load_model(model_path) -> YOLO
    run_segmentation(model, image_path) -> dict

Future Extension:
    This module can be extended to return additional metadata
    (class labels, confidence scores, bounding boxes) that downstream
    modules such as DINOv2 feature extraction can consume.
"""

import os
import logging

import cv2
import numpy as np
from ultralytics import YOLO

from utils.timer import Timer

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> YOLO:
    """
    Load a YOLO segmentation model from disk.

    Args:
        model_path: Absolute or relative path to the .pt weights file.

    Returns:
        A loaded YOLO model instance.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
        RuntimeError: If the model fails to load.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please place your trained best.pt inside the models/ directory."
        )

    try:
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully from '%s'", model_path)
        return model
    except Exception as e:
        logger.exception("Failed to load YOLO model from '%s'", model_path)
        raise RuntimeError(f"Could not load YOLO model: {e}") from e


def run_segmentation(model: YOLO, image_path: str) -> dict:
    """
    Run YOLO segmentation on a single image.

    Args:
        model: A pre-loaded YOLO model instance.
        image_path: Path to the input image.

    Returns:
        A dictionary containing:
            - "segmented_image"  : np.ndarray (BGR) – the image with
              segmentation overlay drawn on it.
            - "mask"             : np.ndarray (uint8, single-channel) –
              binary segmentation mask (255 = vessel, 0 = background).
            - "time"             : float – inference time in seconds.
            - "has_detections"   : bool – whether any objects were detected.
            - "confidence"       : float | None – highest detection confidence.

    Raises:
        ValueError: If the image cannot be read.
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image at '{image_path}'.")

    height, width = original.shape[:2]

    with Timer() as t:
        results = model(image_path, verbose=False)

    result = results[0]

    # ----- No detections -----
    if result.masks is None or len(result.masks) == 0:
        logger.warning("No segmentation masks detected for '%s'.", image_path)
        return {
            "segmented_image": original.copy(),
            "mask": np.zeros((height, width), dtype=np.uint8),
            "time": t.elapsed,
            "has_detections": False,
            "confidence": None,
        }

    # ----- Build combined binary mask from all detected segments -----
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for seg_mask in result.masks.data:
        # Each mask is a tensor on the model's device; move to CPU / numpy
        mask_np = seg_mask.cpu().numpy()

        # Resize mask to the original image dimensions (model may resize internally)
        if mask_np.shape[:2] != (height, width):
            mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)

        combined_mask = cv2.bitwise_or(combined_mask, (mask_np > 0.5).astype(np.uint8) * 255)

    # ----- Create the segmented overlay image -----
    segmented_image = original.copy()
    overlay_color = np.array([0, 200, 0], dtype=np.uint8)  # Green overlay
    alpha = 0.45

    mask_bool = combined_mask > 0
    segmented_image[mask_bool] = cv2.addWeighted(
        segmented_image[mask_bool], 1 - alpha,
        np.full_like(segmented_image[mask_bool], overlay_color), alpha,
        0,
    )

    # Draw contour outlines for clarity
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)

    # ----- Highest confidence score -----
    confidence = None
    if result.boxes is not None and len(result.boxes.conf) > 0:
        confidence = float(result.boxes.conf.max().cpu().item())

    return {
        "segmented_image": segmented_image,
        "mask": combined_mask,
        "time": t.elapsed,
        "has_detections": True,
        "confidence": confidence,
    }
