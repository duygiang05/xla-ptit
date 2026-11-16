"""Tạo mask ứng viên khói theo bối cảnh bằng OpenCV, không dùng mô hình AI."""

import cv2
import numpy as np

from .smoke_detection import local_std


def context_smoke_candidates(
    frame_bgr: np.ndarray,
    fire_mask: np.ndarray,
    *,
    return_aerial_tint: bool = False,
    return_lower: bool = False
):
    """Trả mask TV1 cùng cờ nhận diện cảnh trời; có thể kèm mask phần dưới.

    Cảnh trời cho phép khói chuyển động chậm; cảnh khác cần kiểm tra chuyển
    động ở Fusion. Đây chỉ là ứng viên, chưa phải kết luận có khói.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    height, width = value.shape
    blue = (hue >= 82) & (hue <= 125) & (saturation >= 35) & (value >= 90)
    blue_fraction = float(blue[: height // 2].mean())
    upper_blue_values = value[: height // 2][blue[: height // 2]]
    upper_blue_saturation = saturation[: height // 2][blue[: height // 2]]
    # Ám xanh phủ gần hết ảnh trên cao không chứng minh đó là bầu trời.
    aerial_tint = (
        blue_fraction > 0.9
        and upper_blue_saturation.size > 0
        and float(np.median(upper_blue_saturation)) < 80
    )
    sky_scene = blue_fraction >= 0.17 and not aerial_tint
    strong_blue_sky = (
        sky_scene
        and blue_fraction >= 0.5
        and upper_blue_saturation.size > 0
        and float(np.median(upper_blue_saturation)) >= 80
    )
    bright_sky_floor = (
        max(90, int(np.quantile(upper_blue_values, 0.8)) - 12)
        if upper_blue_values.size
        else 255
    )
    sky = blue & (value >= bright_sky_floor)
    smooth = local_std(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 9) <= 25
    lower_candidates = np.zeros((height, width), dtype=np.uint8)

    if strong_blue_sky:
        # Cột khói xám ít bão hòa hơn trời xanh; tránh gộp cả trời/đồi.
        mask = (saturation <= 80) & (value >= 40) & (value <= 215) & smooth
        mask &= ~sky
    elif sky_scene:
        neutral = (saturation <= 135) & (value >= 45) & (value <= 210)
        warm = (hue <= 33) & (saturation <= 165) & (value >= 45) & (value <= 215)
        mask = (neutral | warm) & ~sky & smooth
        sky_fraction_per_row = sky.mean(axis=1)
        sky_rows = np.flatnonzero(sky_fraction_per_row >= 0.20)
        limit = min(height, int(sky_rows[-1]) + 105) if sky_rows.size else height
        if limit < height:
            lower_candidates[limit:] = mask[limit:].astype(np.uint8) * 255
        mask[limit:] = False
    else:
        bright = (saturation <= 80) & (value >= 130) & (value <= 240)
        mask = bright & smooth

    result = mask.astype(np.uint8) * 255
    # Khói nâu có thể trùng ngưỡng HSV của lửa; chỉ bỏ lõi lửa sáng/bão hòa.
    bright_fire = cv2.bitwise_and(
        fire_mask, ((value >= 190) & (saturation >= 120)).astype(np.uint8) * 255
    )
    fire_expanded = cv2.dilate(
        bright_fire, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    candidates = cv2.bitwise_and(result, cv2.bitwise_not(fire_expanded))
    lower_candidates = cv2.bitwise_and(lower_candidates, cv2.bitwise_not(fire_expanded))
    if return_lower:
        if return_aerial_tint:
            return candidates, sky_scene, aerial_tint, lower_candidates
        return candidates, sky_scene, lower_candidates
    if return_aerial_tint:
        return candidates, sky_scene, aerial_tint
    return candidates, sky_scene
