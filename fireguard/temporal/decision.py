"""TV4: quy tắc thời gian và biến dạng hình học, dễ giải thích.

Các ngưỡng Hybrid là GIẢ THUYẾT cần hiệu chỉnh trên development/integration.
Không gọi kết quả này là xác suất cháy hoặc khói.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ShapeThresholds:
    """Mức biến dạng tối thiểu để Hybrid coi là có dấu hiệu động học.

    Đây là ngưỡng thử nghiệm, không phải ngưỡng tối ưu đã đánh giá.
    """
    fire_area_cv: float = 0.08
    fire_perimeter_cv: float = 0.08
    fire_circularity_std: float = 0.025
    smoke_area_cv: float = 0.08
    smoke_area_growth: float = 0.10
    smoke_centroid_shift: float = 1.5


def passes_temporal_vote(hits: int, required_hits: int) -> bool:
    return hits >= required_hits


def passes_shape_evidence(kind: str, changes: Mapping[str, float],
                          thresholds: ShapeThresholds = ShapeThresholds()) -> bool:
    """Bằng chứng thay đổi hình dạng, sau khi vùng đã đạt Temporal Voting.

    FIRE: dao động diện tích, chu vi hoặc độ tròn.
    SMOKE: biến đổi diện tích, lan rộng hoặc tâm dịch chuyển.
    Không xem dịch chuyển là bằng chứng duy nhất của FIRE vì xe đỏ có thể đi ngang.
    """
    if kind == 'fire':
        return (changes.get('area_cv', 0.0) >= thresholds.fire_area_cv
                or changes.get('perimeter_cv', 0.0) >= thresholds.fire_perimeter_cv
                or changes.get('circularity_std', 0.0) >= thresholds.fire_circularity_std)
    if kind == 'smoke':
        return (changes.get('area_cv', 0.0) >= thresholds.smoke_area_cv
                or changes.get('area_growth', 0.0) >= thresholds.smoke_area_growth
                or changes.get('centroid_shift', 0.0) >= thresholds.smoke_centroid_shift)
    return False
