"""Tính optical flow Farneback để kiểm tra vùng khói."""

import cv2
import numpy as np


def dense_flow_in_bounds(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Tính flow trên phần ảnh thật, bỏ viền đen của video dọc."""
    x, y, w, h = bounds
    return cv2.calcOpticalFlowFarneback(
        previous_gray[y : y + h, x : x + w],
        current_gray[y : y + h, x : x + w],
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def moving_fraction(
    flow: np.ndarray,
    bbox: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
    threshold: float = 0.5,
) -> float:
    """Tỉ lệ pixel chuyển động đủ mạnh trong hộp giới hạn của flow."""
    bx, by, bw, bh = bbox
    x, y, width, height = bounds
    left = max(0, bx - x)
    top = max(0, by - y)
    right = min(width, bx + bw - x)
    bottom = min(height, by + bh - y)
    if right <= left or bottom <= top:
        return 0.0
    roi = flow[top:bottom, left:right]
    squared = roi[:, :, 0] ** 2 + roi[:, :, 1] ** 2
    return float(np.mean(squared >= threshold**2))
