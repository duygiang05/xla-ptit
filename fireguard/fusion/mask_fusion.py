
"""Fusion of TV1 Color and TV2 Motion masks.

Supported modes:
    and           : pixel-wise AND
    and_dilate    : AND with dilated motion
    color_only    : color mask only
    region_motion : confirm color regions using motion evidence

All outputs are binary uint8 masks containing 0 and 255.
"""

import cv2
import numpy as np

from fireguard.shape.morphology import combine_masks


MODES = (
    "and",
    "and_dilate",
    "color_only",
    "region_motion",
)


def _validate_mask(name: str, mask: np.ndarray) -> None:
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.dtype != np.uint8
    ):
        raise TypeError(
            f"{name} must be a 2D numpy uint8 mask"
        )

    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError(
            f"{name} must contain only 0 and 255"
        )


def _region_motion_fusion(
    color_mask: np.ndarray,
    motion_mask: np.ndarray,
    motion_dilate: int,
    region_close: int,
    region_motion_ratio: float,
    region_min_motion_pixels: int,
    region_min_area: int,
) -> np.ndarray:
    """Keep color regions supported by sufficient motion evidence.

    Steps:
        1. Close small gaps in color mask.
        2. Identify connected color components.
        3. Dilate motion mask.
        4. Measure motion evidence inside each color component.
        5. Keep the entire accepted component.

    The returned mask retains color regions rather than just
    their pixel-wise intersection with motion.
    """

    # Close holes and connect nearby smoke-colored pixels.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (region_close, region_close),
    )

    candidate_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    # Allow small position differences between motion and color.
    motion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (motion_dilate, motion_dilate),
    )

    expanded_motion = cv2.dilate(
        motion_mask,
        motion_kernel,
        iterations=1,
    )

    # Label connected candidate regions.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8,
        )
    )

    result = np.zeros_like(color_mask)

    for component_id in range(1, count):
        x, y, w, h, component_area = stats[component_id]

        if component_area < region_min_area:
            continue

        component_roi = (
            labels[y:y + h, x:x + w] == component_id
        )

        motion_roi = (
            expanded_motion[y:y + h, x:x + w] > 0
        )

        motion_pixels = int(
            np.count_nonzero(component_roi & motion_roi)
        )

        motion_ratio = (
            motion_pixels / component_area
        )

        # Confirm motion evidence for the whole color region.
        accepted = (
            motion_pixels >= region_min_motion_pixels
            and motion_ratio >= region_motion_ratio
        )

        if accepted:
            result_roi = result[y:y + h, x:x + w]
            result_roi[component_roi] = 255

    return result


def fuse_masks(
    color_mask: np.ndarray,
    motion_mask: np.ndarray,
    mode: str = "and",
    motion_dilate: int = 5,
    *,
    region_close: int = 15,
    region_motion_ratio: float = 0.015,
    region_min_motion_pixels: int = 12,
    region_min_area: int = 130,
) -> np.ndarray:
    """Combine Color and Motion masks.

    Parameters for region_motion are experimental defaults.
    They must be evaluated using positive and negative videos.
    """

    if mode not in MODES:
        raise ValueError(
            f"Fusion mode must be one of {MODES}, got {mode!r}"
        )

    _validate_mask("color_mask", color_mask)
    _validate_mask("motion_mask", motion_mask)

    if color_mask.shape != motion_mask.shape:
        raise ValueError(
            "Color and motion masks must have the same shape"
        )

    if (
        not isinstance(motion_dilate, int)
        or isinstance(motion_dilate, bool)
        or motion_dilate < 1
    ):
        raise ValueError(
            "motion_dilate must be an integer >= 1"
        )

    # Existing modes: preserve previous behavior.
    if mode == "color_only":
        return combine_masks(
            color_mask,
            motion_mask,
            mode="color",
        )

    if mode == "and":
        return combine_masks(
            color_mask,
            motion_mask,
            mode="and",
            motion_dilate=0,
        )

    if mode == "and_dilate":
        return combine_masks(
            color_mask,
            motion_mask,
            mode="and",
            motion_dilate=motion_dilate,
        )

    # New mode: region-level motion validation.
    if (
        not isinstance(region_close, int)
        or isinstance(region_close, bool)
        or region_close < 1
        or region_close % 2 == 0
    ):
        raise ValueError(
            "region_close must be a positive odd integer"
        )

    if not 0 <= region_motion_ratio <= 1:
        raise ValueError(
            "region_motion_ratio must be between 0 and 1"
        )

    if region_min_motion_pixels < 1:
        raise ValueError(
            "region_min_motion_pixels must be >= 1"
        )

    if region_min_area < 1:
        raise ValueError(
            "region_min_area must be >= 1"
        )

    return _region_motion_fusion(
        color_mask=color_mask,
        motion_mask=motion_mask,
        motion_dilate=motion_dilate,
        region_close=region_close,
        region_motion_ratio=region_motion_ratio,
        region_min_motion_pixels=region_min_motion_pixels,
        region_min_area=region_min_area,
    )
