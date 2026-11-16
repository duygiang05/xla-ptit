"""TV3: xác minh vùng khói theo bối cảnh bằng quy tắc OpenCV."""

import cv2
import numpy as np

from fireguard.motion.flow import moving_fraction
from .regions import Region


def active_image_bounds(frame_bgr):
    """Trả vùng ảnh không đen; nếu ảnh đen hoàn toàn thì trả cả khung."""
    nonblack = np.any(frame_bgr != 0, axis=2)
    rows = np.flatnonzero(np.any(nonblack, axis=1))
    cols = np.flatnonzero(np.any(nonblack, axis=0))
    height, width = frame_bgr.shape[:2]
    if not rows.size or not cols.size:
        return 0, 0, width, height
    return (
        int(cols[0]),
        int(rows[0]),
        int(cols[-1] - cols[0] + 1),
        int(rows[-1] - rows[0] + 1),
    )


def verify_context_regions(
    regions,
    frame_bgr,
    smoke_mask,
    sky_scene,
    *,
    aerial_tint=False,
    motion_mask=None,
    fire_regions=(),
    portrait_flow=None
):
    """Loại vùng khói sai theo hình học, biên, chuyển động và loại cảnh."""
    height, width = smoke_mask.shape
    active_x, active_y, active_width, active_height = active_image_bounds(frame_bgr)
    portrait = active_width < 0.6 * width
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    saturation = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    accepted = []
    for region in regions:
        x, y, w, h = region.bbox
        if region.area < 500 or w <= 0 or h <= 0:
            continue
        if portrait:
            covered_by_fire = 0
            for flame in fire_regions:
                fx, fy, fw, fh = flame.bbox
                covered_by_fire += max(0, min(x + w, fx + fw) - max(x, fx)) * max(
                    0, min(y + h, fy + fh) - max(y, fy)
                )
            if covered_by_fire > 0.25 * w * h:
                continue
        foreground = smoke_mask[y : y + h, x : x + w] > 0
        if not foreground.any():
            continue
        edge_density = float(np.mean(edges[y : y + h, x : x + w][foreground] > 0))
        motion_ratio = 0.0
        if motion_mask is not None:
            motion_ratio = float(
                np.mean(motion_mask[y : y + h, x : x + w][foreground] > 0)
            )
        if w * h > 0.65 * width * height:
            # Cột khói lớn có thể chiếm phần lớn ảnh; tường sáng phủ toàn ảnh
            # như ở barbeq thì không được tính là khói.
            upper_diffuse_plume = (
                not portrait
                and not sky_scene
                and not aerial_tint
                and y <= active_y + 0.05 * active_height
                and y + h < active_y + 0.85 * active_height
                and region.area / (w * h) < 0.55
                and edge_density <= 0.08
                and motion_ratio >= 0.005
            )
            if not upper_diffuse_plume:
                continue
        if aerial_tint:
            # Vùng màu lớn, ít vân trong ảnh trên cao thường do ám màu camera.
            if w * h > 0.35 * width * height:
                continue
            if w < 20 or h < 20 or y + h < 0.55 * height:
                continue
        elif sky_scene:
            if edge_density > 0.16:
                continue
            mean_saturation = float(
                np.mean(saturation[y : y + h, x : x + w][foreground])
            )
            # Mây/nhà tạo dải ngang thấp; cột khói nâu có thể thấp hơn.
            moving_short_plume = (
                portrait and h >= 0.25 * active_height and motion_ratio >= 0.025
            )
            if (
                h < 0.45 * active_height
                and (mean_saturation < 90 or h < 0.25 * active_height)
                and not moving_short_plume
            ):
                continue
            # Dải chân trời mỏng kéo ngang toàn ảnh không phải cột khói.
            if w > 0.8 * active_width and h < 0.35 * active_height:
                continue
            if region.area < 1500:
                continue
        else:
            if edge_density > (0.16 if portrait else 0.08):
                continue
            # Vật rắn chuyển động làm gần hết ROI sáng trên flow; khói mỏng thì không.
            if (
                portrait
                and portrait_flow is not None
                and moving_fraction(
                    portrait_flow,
                    region.bbox,
                    (active_x, active_y, active_width, active_height),
                )
                > 0.95
            ):
                continue
            if portrait and (
                y + h < active_y + 0.55 * active_height
                or (y < active_y + 0.2 * active_height and h > 0.4 * active_height)
            ):
                continue
            # Ưu tiên cột khói rộng ở cảnh mặt đất; có thể bỏ sót cột hẹp.
            if w < 0.4 * active_width or y > active_y + 0.45 * active_height:
                continue
        accepted.append(region)
    accepted.sort(key=lambda item: item.area, reverse=True)
    if not accepted:
        return []
    primary = accepted[0]
    px, py, pw, ph = primary.bbox
    nearby = [primary]
    for region in accepted[1:]:
        x, y, w, h = region.bbox
        if (
            x <= px + pw + 30
            and x + w >= px - 30
            and y <= py + ph + 30
            and y + h >= py - 30
        ):
            nearby.append(region)
    return nearby


def extend_sky_plume(upper_regions, lower_regions, frame_bgr):
    """Nối phần dưới chuyển động, ít cạnh vào vùng khói trời đã xác nhận.

    Phần dưới đã qua TV1 và Fusion TV2; một mình nó không tạo phát hiện mới.
    """
    if not upper_regions or not lower_regions:
        return upper_regions
    active_x, active_y, active_w, active_h = active_image_bounds(frame_bgr)
    portrait = active_w < 0.6 * frame_bgr.shape[1]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    primary = upper_regions[0]
    px, py, pw, ph = primary.bbox
    narrow_upper = pw < 0.25 * active_w
    max_vertical_gap = round(0.22 * active_h) if narrow_upper else 20
    anchors = []
    for lower in lower_regions:
        x, y, w, h = lower.bbox
        max_anchor_width = (
            1.05 if portrait else 0.50 if narrow_upper else 0.45
        ) * active_w
        if (
            lower.area < 1200
            or w > max_anchor_width
            or h < (0.15 if portrait else 0.18) * active_h
            or y < active_y + 0.35 * active_h
        ):
            continue
        if x >= px + pw or x + w <= px:
            continue
        if y > py + ph + max_vertical_gap:
            continue
        edge_density = float(np.mean(edges[y : y + h, x : x + w] > 0))
        if edge_density > (0.10 if portrait else 0.12):
            continue
        anchors.append(lower)
    if not anchors:
        return upper_regions
    if portrait or narrow_upper:
        left = min([px] + [item.bbox[0] for item in anchors])
        right = max([px + pw] + [item.bbox[0] + item.bbox[2] for item in anchors])
        top = min([py] + [item.bbox[1] for item in anchors])
        bottom = max([py + ph] + [item.bbox[1] + item.bbox[3] for item in anchors])
        joined_area = primary.area + sum(item.area for item in anchors)
    else:
        anchor = max(anchors, key=lambda item: item.area)
        x, y, w, h = anchor.bbox
        margin = round(0.10 * active_w)
        left = max(px, x - margin)
        right = min(px + pw, x + w + margin)
        top = min(py, y)
        bottom = max(py + ph, y + h)
        joined_area = primary.area + anchor.area
    if right <= left or bottom <= top:
        return upper_regions
    return [
        Region(
            "smoke",
            (left, top, right - left, bottom - top),
            joined_area,
            ((left + right) / 2, (top + bottom) / 2),
        )
    ]
