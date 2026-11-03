"""
Rim Extraction Module
=====================

Pure OpenCV-based algorithm that computes the **outer rim** of a
segmented vessel from its binary segmentation mask.

This module does NOT use the YOLO model output directly as the rim.
Instead it applies a dedicated image-processing pipeline:

    1. Binarise the segmentation mask.
    2. Keep only the largest connected component (the vessel).
    3. Apply morphological closing to fill small gaps / noise.
    4. Find the external contour  (cv2.RETR_EXTERNAL).
    5. Optionally smooth the contour with cv2.approxPolyDP.
    6. Draw the outer contour at a configurable thickness → rim mask.
    7. Overlay the rim on the original image for visualisation.

Performance target: < 50 ms per image on CPU.

Functions:
    extract_rim(mask, original_image, ...) -> dict

Future Extension:
    The rim contour points returned here can later feed a DINOv2
    feature extractor for vessel identification.
"""

import logging

import cv2
import numpy as np

from utils.timer import Timer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Default parameters (all configurable via function args)
# ──────────────────────────────────────────────────────────
DEFAULT_RIM_THICKNESS: int = 3          # pixels (range 2-5 recommended)
DEFAULT_MORPH_KERNEL_SIZE: int = 15     # kernel size for morphological closing
DEFAULT_MORPH_ITERATIONS: int = 2       # iterations for closing
DEFAULT_SMOOTH_EPSILON_RATIO: float = 0.002  # approxPolyDP epsilon as a
                                              # fraction of contour perimeter
DEFAULT_RIM_COLOR: tuple = (0, 0, 255)  # BGR – red


def _binarise_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert any mask to a clean binary image (0 or 255).

    Handles both float [0..1] and uint8 [0..255] inputs.
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # Ensure single-channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary image.

    This removes small spurious blobs that are not the vessel.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        # Only background – return as-is
        return binary

    # Label 0 is always background; find the largest foreground component
    # stats[:, cv2.CC_STAT_AREA] gives area of each label
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_label = 1 + int(np.argmax(areas))

    cleaned = np.zeros_like(binary)
    cleaned[labels == largest_label] = 255
    return cleaned


def _morphological_close(
    binary: np.ndarray,
    kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
    iterations: int = DEFAULT_MORPH_ITERATIONS,
) -> np.ndarray:
    """
    Apply morphological closing to fill small gaps and smooth edges.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )
    return closed


def _find_outer_contour(binary: np.ndarray) -> np.ndarray | None:
    """
    Find the single largest external contour in the binary image.

    Returns:
        The contour (np.ndarray) or None if no contour is found.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Pick the contour with the largest area
    largest = max(contours, key=cv2.contourArea)
    return largest


def _smooth_contour(
    contour: np.ndarray,
    epsilon_ratio: float = DEFAULT_SMOOTH_EPSILON_RATIO,
) -> np.ndarray:
    """
    Smooth / simplify the contour using the Douglas-Peucker algorithm.

    Args:
        contour: Input contour points.
        epsilon_ratio: Approximation accuracy as a fraction of the
                       contour's arc length.  Smaller = more faithful.

    Returns:
        Smoothed contour.
    """
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_ratio * perimeter
    smoothed = cv2.approxPolyDP(contour, epsilon, closed=True)
    return smoothed


def extract_rim(
    mask: np.ndarray,
    original_image: np.ndarray,
    rim_thickness: int = DEFAULT_RIM_THICKNESS,
    rim_color: tuple = DEFAULT_RIM_COLOR,
    morph_kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
    morph_iterations: int = DEFAULT_MORPH_ITERATIONS,
    smooth_epsilon_ratio: float = DEFAULT_SMOOTH_EPSILON_RATIO,
    smooth_contour: bool = True,
) -> dict:
    """
    Extract the outer rim of a vessel from a segmentation mask.

    Pipeline:
        binarise → largest component → morph close →
        external contour → optional smooth → draw rim

    Args:
        mask:                  Segmentation mask (uint8 or float).
        original_image:        Original BGR image for overlay.
        rim_thickness:         Contour line thickness in pixels (2-5).
        rim_color:             BGR colour tuple for the rim line.
        morph_kernel_size:     Kernel size for morphological closing.
        morph_iterations:      Number of closing iterations.
        smooth_epsilon_ratio:  Smoothing factor for approxPolyDP.
        smooth_contour:        Whether to apply contour smoothing.

    Returns:
        dict with keys:
            - "rim_image"      : np.ndarray – original image with rim overlay.
            - "rim_mask"       : np.ndarray – binary mask of the rim only.
            - "rim_on_black"   : np.ndarray – rim drawn on a black background.
            - "contour"        : np.ndarray | None – the rim contour points.
            - "time"           : float – extraction time in seconds.
            - "has_rim"        : bool – whether a valid rim was found.
    """
    height, width = original_image.shape[:2]

    with Timer() as t:
        # ── Step 1: Binarise ──────────────────────────────────────
        binary = _binarise_mask(mask)

        # ── Step 2: Largest connected component ───────────────────
        binary = _keep_largest_component(binary)

        # ── Step 3: Morphological closing ─────────────────────────
        binary = _morphological_close(
            binary,
            kernel_size=morph_kernel_size,
            iterations=morph_iterations,
        )

        # ── Step 4: External contour ─────────────────────────────
        contour = _find_outer_contour(binary)

        if contour is None:
            logger.warning("No outer contour found during rim extraction.")
            return {
                "rim_image": original_image.copy(),
                "rim_mask": np.zeros((height, width), dtype=np.uint8),
                "rim_on_black": np.zeros_like(original_image),
                "contour": None,
                "time": t.elapsed,
                "has_rim": False,
            }

        # ── Step 5: Optional contour smoothing ───────────────────
        if smooth_contour:
            contour = _smooth_contour(contour, epsilon_ratio=smooth_epsilon_ratio)

        # ── Step 6: Draw the rim ─────────────────────────────────
        # 6a. Rim mask (white contour on black)
        rim_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(rim_mask, [contour], -1, 255, rim_thickness)

        # 6b. Rim on black background (coloured)
        rim_on_black = np.zeros_like(original_image)
        cv2.drawContours(rim_on_black, [contour], -1, rim_color, rim_thickness)

        # 6c. Rim overlaid on original image
        rim_image = original_image.copy()
        cv2.drawContours(rim_image, [contour], -1, rim_color, rim_thickness)

    logger.info("Rim extraction completed in %.4f sec.", t.elapsed)

    return {
        "rim_image": rim_image,
        "rim_mask": rim_mask,
        "rim_on_black": rim_on_black,
        "contour": contour,
        "time": t.elapsed,
        "has_rim": True,
    }
