"""Ghép mask màu TV1 với mask chuyển động TV2, đầu ra nhị phân 0/255.

Các chế độ: giao pixel, giao sau giãn, chỉ màu hoặc xác nhận theo vùng.
"""

import cv2
import numpy as np

from fireguard.shape.morphology import combine_masks

MODES = ("and", "and_dilate", "color_only", "region_motion")


def _validate_mask(name: str, mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise TypeError(f"{name} must be a 2D numpy uint8 mask")

    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError(f"{name} must contain only 0 and 255")


def _region_motion_fusion(
    color_mask: np.ndarray,
    motion_mask: np.ndarray,
    motion_dilate: int,
    region_close: int,
    region_motion_ratio: float,
    region_min_motion_pixels: int,
    region_min_area: int,
) -> np.ndarray:
    """Giữ toàn vùng màu nếu đủ bằng chứng chuyển động trong vùng đó.

    Đóng khe nhỏ, tìm thành phần liên thông, giãn mask chuyển động rồi
    kiểm tra số/tỉ lệ pixel động. Kết quả giữ cả vùng, không chỉ phần giao.
    """

    # Nối các mảng ứng viên màu gần nhau.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (region_close, region_close),
    )

    candidate_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel)

    # Bù lệch vị trí nhỏ giữa màu và chuyển động.
    motion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (motion_dilate, motion_dilate),
    )

    expanded_motion = cv2.dilate(motion_mask, motion_kernel, iterations=1)

    # Gán nhãn các thành phần liên thông.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )

    result = np.zeros_like(color_mask)

    for component_id in range(1, count):
        x, y, w, h, component_area = stats[component_id]

        if component_area < region_min_area:
            continue

        component_roi = labels[y : y + h, x : x + w] == component_id

        motion_roi = expanded_motion[y : y + h, x : x + w] > 0

        motion_pixels = int(np.count_nonzero(component_roi & motion_roi))

        motion_ratio = motion_pixels / component_area

        # Xác nhận chuyển động trên toàn vùng màu.
        accepted = (
            motion_pixels >= region_min_motion_pixels
            and motion_ratio >= region_motion_ratio
        )

        if accepted:
            result_roi = result[y : y + h, x : x + w]
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
    """Ghép hai mask theo chế độ đã chọn.

    Ngưỡng của `region_motion` cần được kiểm tra trên video dương/âm tính.
    """

    if mode not in MODES:
        raise ValueError(f"Fusion mode must be one of {MODES}, got {mode!r}")

    _validate_mask("color_mask", color_mask)
    _validate_mask("motion_mask", motion_mask)

    if color_mask.shape != motion_mask.shape:
        raise ValueError("Color and motion masks must have the same shape")

    if (
        not isinstance(motion_dilate, int)
        or isinstance(motion_dilate, bool)
        or motion_dilate < 1
    ):
        raise ValueError("motion_dilate must be an integer >= 1")

    # Các chế độ cũ giữ nguyên hành vi.
    if mode == "color_only":
        return combine_masks(color_mask, motion_mask, mode="color")

    if mode == "and":
        return combine_masks(color_mask, motion_mask, mode="and", motion_dilate=0)

    if mode == "and_dilate":
        return combine_masks(
            color_mask,
            motion_mask,
            mode="and",
            motion_dilate=motion_dilate,
        )

    # Kiểm tra tham số cho chế độ xác nhận theo vùng.
    if (
        not isinstance(region_close, int)
        or isinstance(region_close, bool)
        or region_close < 1
        or region_close % 2 == 0
    ):
        raise ValueError("region_close must be a positive odd integer")

    if not 0 <= region_motion_ratio <= 1:
        raise ValueError("region_motion_ratio must be between 0 and 1")

    if region_min_motion_pixels < 1:
        raise ValueError("region_min_motion_pixels must be >= 1")

    if region_min_area < 1:
        raise ValueError("region_min_area must be >= 1")

    return _region_motion_fusion(
        color_mask=color_mask,
        motion_mask=motion_mask,
        motion_dilate=motion_dilate,
        region_close=region_close,
        region_motion_ratio=region_motion_ratio,
        region_min_motion_pixels=region_min_motion_pixels,
        region_min_area=region_min_area,
    )
