"""TV4: các đặc trưng biến đổi qua các frame của cùng một Region.

Nguồn đặc trưng hình dạng đầy đủ: TV3 Track.shape_change().
Các hàm dưới đây giữ tương thích với tests và thực nghiệm TV4 cũ.
"""
from math import hypot
from fireguard.shape.regions import Region
from fireguard.shape.tracking import bbox_iou


def same_candidate(current: Region, previous: Region, cfg) -> bool:
    """Kiểm tra hai vùng có thể ghép (cùng nhãn và gần nhau)."""
    if current.kind != previous.kind:
        return False
    distance = hypot(current.centroid[0] - previous.centroid[0],
                     current.centroid[1] - previous.centroid[1])
    return (bbox_iou(current.bbox, previous.bbox) >= cfg.min_iou
            or distance <= cfg.max_center_distance)


def calculate_temporal_features(previous: Region, current: Region) -> dict[str, float]:
    """Đặc trưng biến động giữa HAI vùng đã được liên kết đúng.

    Thêm thay đổi chu vi/độ tròn khi TV3 đã tính những đại lượng này.
    """
    if previous.kind != current.kind:
        raise ValueError('Không so sánh hai vùng khác nhãn FIRE/SMOKE')
    old_ratio = previous.bbox[2] / max(previous.bbox[3], 1)
    new_ratio = current.bbox[2] / max(current.bbox[3], 1)
    out = {
        'area_change': (current.area - previous.area) / max(previous.area, 1.0),
        'centroid_displacement': hypot(current.centroid[0] - previous.centroid[0],
                                       current.centroid[1] - previous.centroid[1]),
        'aspect_ratio_change': (new_ratio - old_ratio) / max(old_ratio, 1e-6),
    }
    if previous.perimeter > 0 and current.perimeter > 0:
        out['perimeter_change'] = (current.perimeter - previous.perimeter) / previous.perimeter
    if previous.circularity > 0 and current.circularity > 0:
        out['circularity_change'] = current.circularity - previous.circularity
    return out
