"""TV3 – Đặc trưng hình dạng của một vùng (contour).

Các đặc trưng đều bất biến tỉ lệ (trừ area/perimeter) để TV4 so sánh qua frame:
- perimeter   : chu vi contour, cv2.arcLength(closed=True).
- circularity : 4*pi*A / P^2, bằng 1 với hình tròn; lửa/khói có biên răng cưa nên thấp.
- solidity    : A / A_convex_hull, vùng lồi lõm (ngọn lửa) có solidity nhỏ.
- aspect_ratio: w / h của bounding box thẳng.
- extent      : A / (w*h), mức độ vùng lấp đầy bounding box.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapeFeatures:
    area: float
    perimeter: float
    circularity: float
    solidity: float
    aspect_ratio: float
    extent: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def shape_features(contour: np.ndarray) -> ShapeFeatures:
    """Tính đặc trưng hình dạng cho một contour của cv2.findContours."""
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    m = cv2.moments(contour)
    if m["m00"]:
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    else:
        centroid = (x + w / 2.0, y + h / 2.0)
    return ShapeFeatures(
        area=area,
        perimeter=perimeter,
        circularity=min(1.0, _safe_div(4.0 * math.pi * area, perimeter * perimeter)),
        solidity=min(1.0, _safe_div(area, hull_area)),
        aspect_ratio=_safe_div(w, h),
        extent=min(1.0, _safe_div(area, w * h)),
        bbox=(int(x), int(y), int(w), int(h)),
        centroid=(float(centroid[0]), float(centroid[1])),
    )
