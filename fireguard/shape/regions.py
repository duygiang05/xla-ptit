"""TV3 – Trích vùng liên thông và đặc trưng vùng từ mask nhị phân."""
from dataclasses import dataclass

import cv2
import numpy as np

from fireguard.config import Config
from .features import shape_features
from .morphology import clean_mask


@dataclass(frozen=True)
class Region:
    # 5 trường đầu giữ nguyên thứ tự theo INTERFACES.md; các trường sau có mặc định
    # nên code cũ tạo Region(kind, bbox, area, centroid[, score]) vẫn chạy.
    kind: str
    bbox: tuple[int, int, int, int]
    area: float
    centroid: tuple[float, float]
    score: float = 1.0
    perimeter: float = 0.0
    circularity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    extent: float = 0.0


def _region_from_contour(contour: np.ndarray, kind: str) -> Region:
    f = shape_features(contour)
    return Region(kind, f.bbox, f.area, f.centroid, 1.0,
                  f.perimeter, f.circularity, f.solidity, f.aspect_ratio, f.extent)


def extract_regions(mask, kind: str, cfg: Config, clean: bool = True) -> list[Region]:
    """Mask -> (morphology) -> findContours -> danh sách Region có đặc trưng hình dạng.

    clean=False bỏ qua bước morphology (dùng cho thí nghiệm có/không morphology).
    """
    if clean:
        mask = clean_mask(mask, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for contour in contours:
        if cv2.contourArea(contour) < cfg.min_region_area:
            continue
        items.append(_region_from_contour(contour, kind))
    return items


def extract_regions_cc(mask, kind: str, cfg: Config, clean: bool = True,
                       connectivity: int = 8) -> list[Region]:
    """Phương án so sánh dùng cv2.connectedComponentsWithStats.

    Diện tích ở đây là số pixel của thành phần (khác contourArea tính theo đa giác
    biên), bbox/centroid lấy trực tiếp từ stats; chu vi và đặc trưng hình dạng
    tính từ contour ngoài của từng thành phần.
    """
    if clean:
        mask = clean_mask(mask, cfg)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=connectivity)
    items = []
    for label in range(1, n):  # 0 là nền
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < cfg.min_region_area:
            continue
        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        f = shape_features(max(contours, key=cv2.contourArea))
        cx, cy = centroids[label]
        items.append(Region(kind, (x, y, w, h), float(area), (float(cx), float(cy)), 1.0,
                            f.perimeter, f.circularity, f.solidity, f.aspect_ratio, f.extent))
    return items


def draw_regions(frame_bgr: np.ndarray, regions: list[Region], show_contour_info: bool = True,
                 color_map: dict[str, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Vẽ bbox + tâm + (tùy chọn) area/circularity để làm ảnh minh họa báo cáo."""
    color_map = color_map or {"fire": (0, 0, 255), "smoke": (255, 200, 0)}
    out = frame_bgr.copy()
    for r in regions:
        x, y, w, h = r.bbox
        col = color_map.get(r.kind, (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.circle(out, (int(r.centroid[0]), int(r.centroid[1])), 3, col, -1)
        if show_contour_info:
            label = f"{r.kind} A={r.area:.0f} C={r.circularity:.2f}"
            cv2.putText(out, label, (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return out
