"""Fusion Color (TV1) + Motion (TV2), tái sử dụng combine_masks của TV3.

Mọi mode đều trả về uint8 2D 0/255, kích thước giống color_mask.
"""
import numpy as np
from fireguard.shape.morphology import combine_masks

MODES = ("and", "and_dilate", "color_only")


def fuse_masks(color_mask: np.ndarray, motion_mask: np.ndarray,
               mode: str = "and", motion_dilate: int = 5) -> np.ndarray:
    """Chế độ baseline AND, AND + giãn motion, hoặc chỉ color (ablation).

    `motion_dilate` là kích thước kernel như API TV3 (5 = 5x5),
    KHÔNG phải bán kính 5 pixel.
    """
    if mode not in MODES:
        raise ValueError(f"Fusion mode phải thuộc {MODES}; nhận được {mode!r}")
    for name, mask in (("color_mask", color_mask), ("motion_mask", motion_mask)):
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
            raise TypeError(f"{name} phải là numpy uint8 mask 2 chiều")
        if not np.all((mask == 0) | (mask == 255)):
            raise ValueError(f"{name} chỉ được chứa 0/255")
    if color_mask.shape != motion_mask.shape:
        raise ValueError("Hai mask phải cùng kích thước")
    if not isinstance(motion_dilate, int) or isinstance(motion_dilate, bool) or motion_dilate < 1:
        raise ValueError("motion_dilate phải là số nguyên >= 1")
    if mode == "color_only":
        return combine_masks(color_mask, motion_mask, mode="color")
    if mode == "and":
        return combine_masks(color_mask, motion_mask, mode="and", motion_dilate=0)
    return combine_masks(color_mask, motion_mask, mode="and", motion_dilate=motion_dilate)
