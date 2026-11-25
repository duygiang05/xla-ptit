"""Ghép mask màu TV1 với mask chuyển động TV2, đầu ra nhị phân 0/255.

Các chế độ: giao pixel, giao sau giãn, chỉ màu, khói trời hoặc xác nhận theo vùng.
"""

import cv2
import numpy as np

from fireguard.shape.morphology import combine_masks

MODES = ("and", "and_dilate", "color_only", "sky_motion", "region_motion", "adaptive_region_motion")


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
    region_max_area_ratio: float | None = None,
    growth_kernel: int = 5,
    growth_iterations: int = 8,
) -> np.ndarray:
    """Giữ toàn vùng màu nếu đủ bằng chứng chuyển động trong vùng đó.

    Đóng khe nhỏ, tìm thành phần liên thông, giãn mask chuyển động rồi
    kiểm tra số/tỉ lệ pixel động. Kết quả giữ cả vùng, không chỉ phần giao.
    Adaptive: vùng quá lớn chỉ phát triển từ giao màu/chuyển động hợp lệ,
    giới hạn số lần giãn và không vượt ra ngoài thành phần màu.
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
    growth_element = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (growth_kernel, growth_kernel))
        if region_max_area_ratio is not None and growth_iterations > 0
        else None
    )

    for component_id in range(1, count):
        x, y, w, h, component_area = stats[component_id]

        if component_area < region_min_area:
            continue

        component_roi = labels[y : y + h, x : x + w] == component_id

        motion_roi = expanded_motion[y : y + h, x : x + w] > 0

        motion_pixels = int(np.count_nonzero(component_roi & motion_roi))

        if (
            region_max_area_ratio is not None
            and component_area / color_mask.size > region_max_area_ratio
        ):
            if motion_pixels < region_min_motion_pixels:
                continue
            intersection = (component_roi & motion_roi).astype(np.uint8)
            intersection_count, intersection_labels, intersection_stats, _ = (
                cv2.connectedComponentsWithStats(intersection, connectivity=8)
            )
            # Giữ hạt giống đủ diện tích như Adaptive V1, trước khi phát triển.
            seed = np.zeros_like(intersection)
            for intersection_id in range(1, intersection_count):
                if intersection_stats[intersection_id, cv2.CC_STAT_AREA] >= region_min_area:
                    seed[intersection_labels == intersection_id] = 255

            # R(k+1) = dilate(R(k)) AND C_i; không lan ra ngoài vùng màu.
            # growth_iterations = 0 giữ fallback V1 để thực nghiệm A/B.
            if growth_element is not None and cv2.countNonZero(seed):
                component_mask = component_roi.astype(np.uint8) * 255
                for _ in range(growth_iterations):
                    grown = cv2.bitwise_and(
                        cv2.dilate(seed, growth_element, iterations=1), component_mask
                    )
                    if np.array_equal(grown, seed):
                        break
                    seed = grown

            result_roi = result[y : y + h, x : x + w]
            result_roi[seed > 0] = 255
            continue

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
    region_max_area_ratio: float = 0.50,
    active_bounds: tuple[int, int, int, int] | None = None,
    growth_kernel: int = 5,
    growth_iterations: int = 8,
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

    if mode == "sky_motion":
        # MOG2 chỉ bắt được vài điểm trên khói chậm. Mở rộng các điểm đó,
        # nối những mảnh gần nhau, nhưng luôn giữ kết quả trong mask màu.
        support = combine_masks(
            color_mask, motion_mask, mode="and", motion_dilate=motion_dilate
        )
        connected = cv2.morphologyEx(
            support,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        return cv2.bitwise_and(connected, color_mask)

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

    if mode == "adaptive_region_motion" and not (
        isinstance(region_max_area_ratio, (int, float))
        and not isinstance(region_max_area_ratio, bool)
        and 0 < region_max_area_ratio <= 1
    ):
        raise ValueError("region_max_area_ratio must be in (0, 1]")

    # Chỉ Adaptive V2 sử dụng vùng ảnh thật và tham số phát triển vùng.
    # Các chế độ cũ không thay đổi hành vi.
    if mode == "adaptive_region_motion":
        if type(growth_kernel) is not int or growth_kernel < 1 or growth_kernel % 2 == 0:
            raise ValueError("growth_kernel must be a positive odd integer")
        if type(growth_iterations) is not int or growth_iterations < 0:
            raise ValueError("growth_iterations must be a non-negative integer")
        if active_bounds is not None:
            x, y, w, h = active_bounds
            height, width = color_mask.shape
            if not (
                all(isinstance(v, (int, np.integer)) for v in (x, y, w, h))
                and x >= 0 and y >= 0 and w > 0 and h > 0
                and x + w <= width and y + h <= height
            ):
                raise ValueError("active_bounds must lie inside the masks")
            # Crop trước Closing để diện tích C_i và mẫu số đều thuộc ảnh thật.
            roi = np.s_[y : y + h, x : x + w]
            result = np.zeros_like(color_mask)
            result[roi] = _region_motion_fusion(
                color_mask=color_mask[roi],
                motion_mask=motion_mask[roi],
                motion_dilate=motion_dilate,
                region_close=region_close,
                region_motion_ratio=region_motion_ratio,
                region_min_motion_pixels=region_min_motion_pixels,
                region_min_area=region_min_area,
                region_max_area_ratio=region_max_area_ratio,
                growth_kernel=growth_kernel,
                growth_iterations=growth_iterations,
            )
            return result

    return _region_motion_fusion(
        color_mask=color_mask,
        motion_mask=motion_mask,
        motion_dilate=motion_dilate,
        region_close=region_close,
        region_motion_ratio=region_motion_ratio,
        region_min_motion_pixels=region_min_motion_pixels,
        region_min_area=region_min_area,
        region_max_area_ratio=(
            region_max_area_ratio if mode == "adaptive_region_motion" else None
        ),
        growth_kernel=growth_kernel,
        growth_iterations=growth_iterations,
    )
