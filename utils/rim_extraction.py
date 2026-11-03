"""
Rim Extraction Module
=====================

Pure OpenCV-based algorithm that computes the **outer rim** of a
segmented vessel.  The rim is derived by subtracting an eroded mask
from the cleaned vessel mask, yielding only the outermost boundary
pixels of the vessel.

This module does NOT use the YOLO model to draw the rim.

Pipeline
--------
    1.  Binarise the segmentation mask.
    2.  Remove connected components that touch the image border.
    3.  Remove small noise regions (food particles / specks).
    4.  Keep the main vessel contour (largest component).
    5.  Fill internal holes (food / inner vessel area).
    6.  Morphological closing so the boundary is continuous.
    7.  Erode the clean mask with an elliptical kernel.
    8.  Compute:  rim = clean_mask - eroded_mask.
    9.  Return the rim as a B&W mask and as the actual rim pixels
        cropped from the original image.

Kernel-size guide
-----------------
    3x3  -> thin rim   (~1-2 px boundary)
    5x5  -> medium rim (~2-3 px boundary)
    9x9  -> thick rim  (~4-5 px boundary)

Performance target: < 50 ms per image on CPU.

Functions
---------
    extract_outer_rim(mask, ...) -> np.ndarray
    extract_rim(mask, original_image=None, ...) -> dict

Future Extension
----------------
    The rim mask returned here can later feed a DINOv2 feature
    extractor for registered vessel identification.
"""

import logging

import cv2
import numpy as np

from utils.timer import Timer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Defaults (all overridable via function arguments)
# ──────────────────────────────────────────────────────────
DEFAULT_EROSION_KERNEL_SIZE: int = 0          # 0 = auto (scales with vessel size)
DEFAULT_CLOSING_KERNEL_SIZE: int = 5          # light smoothing only
DEFAULT_CLOSING_ITERATIONS: int = 1
DEFAULT_MIN_COMPONENT_AREA: int = 500         # px² – ignore blobs smaller than this
DEFAULT_RIM_COLOR: tuple = (255, 255, 255)    # BGR white for the B&W mask
AUTO_RIM_WIDTH_RATIO: float = 0.06            # band ≈ 3% of vessel diameter


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def _binarise(mask: np.ndarray) -> np.ndarray:
    """Convert any mask (float or uint8, 1-ch or 3-ch) to binary 0/255."""
    mask = np.asarray(mask)

    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.dtype != np.uint8:
        if mask.size and float(mask.max()) <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    elif mask.size and int(mask.max()) <= 1:
        mask = (mask * 255).astype(np.uint8)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _remove_border_components(binary: np.ndarray) -> np.ndarray:
    """
    Remove any connected component that touches the image border.

    Border-touching blobs are typically partial objects or background
    artefacts, not the centred vessel.
    """
    h, w = binary.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for label_id in range(1, num_labels):  # skip background (0)
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]

        touches_left = (x == 0)
        touches_top = (y == 0)
        touches_right = (x + bw >= w)
        touches_bottom = (y + bh >= h)

        if touches_left or touches_top or touches_right or touches_bottom:
            continue  # discard this component

        cleaned[labels == label_id] = 255

    return cleaned


def _remove_small_components(
    binary: np.ndarray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
) -> np.ndarray:
    """Remove connected components smaller than *min_area* pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned


def _select_main_vessel(binary: np.ndarray) -> np.ndarray:
    """
    Keep only the single main vessel component.

    Selection heuristic:
        • Primary: largest area (most likely the vessel).
        • Tie-break: closest centroid to image centre.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return binary  # nothing to select

    h, w = binary.shape
    img_cx, img_cy = w / 2.0, h / 2.0

    best_label = -1
    best_key = None

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]
        dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5

        # Larger area wins; smaller distance to centre breaks ties.
        key = (area, -dist)
        if best_key is None or key > best_key:
            best_key = key
            best_label = label_id

    result = np.zeros_like(binary)
    if best_label > 0:
        result[labels == best_label] = 255
    return result


def _morphological_close(
    binary: np.ndarray,
    kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
    iterations: int = DEFAULT_CLOSING_ITERATIONS,
) -> np.ndarray:
    """Apply light morphological closing to smooth edges and fill tiny gaps."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """
    Fill internal holes so only the outer vessel boundary remains.

    Without this, eroding a mask that has holes would also trace the
    inner boundaries (food / inner rim), which must not appear in the
    final outer-rim output.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _resolve_erosion_kernel_size(
    clean_mask: np.ndarray,
    erosion_kernel_size: int,
) -> int:
    """
    Resolve the erosion kernel size.

    If *erosion_kernel_size* is a positive value it is used directly.
    Otherwise the rim width is chosen automatically as a small fraction
    of the vessel's bounding box, so the extracted band corresponds to
    the visible rim instead of a 1-pixel line.
    """
    if erosion_kernel_size and int(erosion_kernel_size) >= 3:
        size = int(erosion_kernel_size)
    else:
        ys, xs = np.where(clean_mask > 0)
        if xs.size == 0:
            return 3
        vessel_size = min(
            int(xs.max() - xs.min()) + 1,
            int(ys.max() - ys.min()) + 1,
        )
        size = max(9, int(round(vessel_size * AUTO_RIM_WIDTH_RATIO)))

    # Elliptical kernels work best with odd dimensions.
    if size % 2 == 0:
        size += 1
    return size


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def extract_outer_rim(
    mask: np.ndarray,
    erosion_kernel_size: int = DEFAULT_EROSION_KERNEL_SIZE,
    closing_kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
    closing_iterations: int = DEFAULT_CLOSING_ITERATIONS,
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
) -> np.ndarray:
    """
    Extract only the outer rim of a vessel from a segmentation mask.

    Algorithm:
        1. Binarise
        2. Remove border-touching components
        3. Remove small noise blobs
        4. Keep the main vessel component
        5. Fill internal holes (food / inner area)
        6. Morphological closing (smooth, continuous boundary)
        7. Erode with an elliptical kernel
        8. rim = clean_mask - eroded_mask

    Args:
        mask:                 Segmentation mask from YOLO (any dtype).
        erosion_kernel_size:  Controls rim thickness. 0 = auto (scales
                              with the vessel size). Explicit values:
                              3->thin, 5->medium, 9->thick, 21->wide band.
        closing_kernel_size:  Kernel for morphological closing.
        closing_iterations:   Iterations for closing.
        min_component_area:   Minimum area (px²) to keep a component.

    Returns:
        np.ndarray: Binary rim mask (uint8, 0/255) containing only
                    the outer boundary pixels of the vessel.
    """
    # Step 1 - Binarise
    binary = _binarise(mask)
    if not np.any(binary):
        return binary

    # Step 2 - Remove border-touching components
    binary = _remove_border_components(binary)

    # Step 3 - Remove small noise
    binary = _remove_small_components(binary, min_area=min_component_area)

    # Step 4 - Select the main vessel
    clean_mask = _select_main_vessel(binary)
    if not np.any(clean_mask):
        return clean_mask

    # Step 5 - Fill internal holes so inner boundaries (food) disappear
    clean_mask = _fill_holes(clean_mask)

    # Step 6 - Light morphological closing to smooth the boundary
    clean_mask = _morphological_close(
        clean_mask,
        kernel_size=closing_kernel_size,
        iterations=closing_iterations,
    )

    # Step 7 - Erode the clean mask inward
    kernel_size = _resolve_erosion_kernel_size(clean_mask, erosion_kernel_size)
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    eroded_mask = cv2.erode(clean_mask, erosion_kernel, iterations=1)

    # Step 8 - Rim = original clean mask - eroded mask
    rim_mask = cv2.subtract(clean_mask, eroded_mask)

    return rim_mask


def extract_rim(
    mask: np.ndarray,
    original_image: np.ndarray = None,
    erosion_kernel_size: int = DEFAULT_EROSION_KERNEL_SIZE,
    closing_kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
    closing_iterations: int = DEFAULT_CLOSING_ITERATIONS,
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
    rim_color: tuple = DEFAULT_RIM_COLOR,
) -> dict:
    """
    Full rim-extraction pipeline with timing and output formats.

    Args:
        mask:                 YOLO segmentation mask.
        original_image:       Optional original BGR image. The real rim
                              pixels are copied from this image.
        erosion_kernel_size:  Controls rim thickness. 0 = auto (scales
                              with the vessel size). Explicit values:
                              3->thin, 5->medium, 9->thick, 21->wide band.
        closing_kernel_size:  Morphological closing kernel.
        closing_iterations:   Closing iterations.
        min_component_area:   Minimum area to keep a component.
        rim_color:            BGR colour for the B&W mask pixels.

    Returns:
        dict with keys:
            "rim_mask"              - binary mask (uint8, 0/255) of the rim.
            "rim_image"             - BGR: white rim on a black background.
            "rim_transparent"       - BGRA: white rim, transparent background.
            "rim_photo"             - BGR: actual rim pixels from the
                                      original image, rest is black.
            "rim_photo_white"       - BGR: actual rim pixels on a white
                                      background (matches reference style).
            "rim_photo_transparent" - BGRA: actual rim pixels with a
                                      transparent background.
            "time"                  - float: extraction time in seconds.
            "has_rim"               - bool: whether a valid rim was found.
    """
    if original_image is not None:
        height, width = original_image.shape[:2]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
    else:
        height, width = mask.shape[:2]

    with Timer() as t:
        rim_mask = extract_outer_rim(
            mask,
            erosion_kernel_size=erosion_kernel_size,
            closing_kernel_size=closing_kernel_size,
            closing_iterations=closing_iterations,
            min_component_area=min_component_area,
        )

    has_rim = bool(np.any(rim_mask > 0))

    # ── Black-and-white rim mask (white rim on black) ──
    rim_image = np.zeros((height, width, 3), dtype=np.uint8)
    rim_transparent = np.zeros((height, width, 4), dtype=np.uint8)

    # ── Actual rim image: real pixels from the original photo ──
    rim_photo = np.zeros((height, width, 3), dtype=np.uint8)
    rim_photo_white = np.full((height, width, 3), 255, dtype=np.uint8)
    rim_photo_transparent = np.zeros((height, width, 4), dtype=np.uint8)

    if has_rim:
        # B&W mask output
        rim_image[rim_mask > 0] = rim_color
        rim_transparent[rim_mask > 0, :3] = rim_color
        rim_transparent[rim_mask > 0, 3] = 255

        # Real-photo rim outputs
        if original_image is not None:
            rim_photo[rim_mask > 0] = original_image[rim_mask > 0]
            rim_photo_white[rim_mask > 0] = original_image[rim_mask > 0]
            rim_photo_transparent[rim_mask > 0, :3] = original_image[rim_mask > 0]
        else:
            rim_photo[rim_mask > 0] = rim_color
            rim_photo_white[rim_mask > 0] = rim_color
            rim_photo_transparent[rim_mask > 0, :3] = rim_color

        rim_photo_transparent[rim_mask > 0, 3] = 255
    else:
        logger.warning("No rim could be extracted from the segmentation mask.")

    logger.info(
        "Rim extraction completed in %.4f sec  (kernel=%s, rim_pixels=%d).",
        t.elapsed,
        "auto" if not erosion_kernel_size else erosion_kernel_size,
        int(np.sum(rim_mask > 0)),
    )

    return {
        "rim_mask": rim_mask,
        "rim_image": rim_image,
        "rim_transparent": rim_transparent,
        "rim_photo": rim_photo,
        "rim_photo_white": rim_photo_white,
        "rim_photo_transparent": rim_photo_transparent,
        "time": t.elapsed,
        "has_rim": has_rim,
    }
