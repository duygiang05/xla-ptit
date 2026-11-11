"""TV3 – Liên kết vùng giữa các frame (KHÔNG phải multi-object tracker đầy đủ).

Ghép tham lam theo IoU bbox, nếu không đủ IoU thì theo khoảng cách tâm gần nhất,
chỉ ghép vùng cùng `kind`. Mỗi track giữ lịch sử Region trong cửa sổ ngắn để
TV4 tính biến thiên diện tích/chu vi/tâm/hình dạng theo thời gian.
"""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Helper for basic association; NOT a full multi-object tracker."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by); x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / float(union) if union > 0 else 0.0


def center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Track:
    track_id: int
    kind: str
    history: deque = field(default_factory=deque)  # deque[(frame_idx, Region)]
    missed: int = 0

    @property
    def last(self):
        return self.history[-1][1]

    @property
    def length(self) -> int:
        return len(self.history)

    def shape_change(self) -> dict[str, float]:
        """Thống kê biến đổi hình dạng trong cửa sổ lịch sử.

        area_cv          : hệ số biến thiên diện tích (std/mean) – lửa/khói dao động,
                           đèn/vật đứng yên gần 0.
        area_growth      : (A_cuối - A_đầu) / A_đầu – khói thường lan rộng dần.
        perimeter_cv     : hệ số biến thiên chu vi.
        circularity_std  : độ lệch chuẩn độ tròn – biên lửa thay đổi liên tục.
        centroid_shift   : quãng đường trung bình tâm di chuyển mỗi bước (pixel).
        bbox_iou_mean    : IoU trung bình giữa hai bước liên tiếp.
        """
        regions = [r for _, r in self.history]
        areas = np.array([r.area for r in regions], dtype=float)
        perims = np.array([r.perimeter for r in regions], dtype=float)
        circ = np.array([r.circularity for r in regions], dtype=float)
        steps = list(zip(regions[:-1], regions[1:]))

        def cv(v):
            return float(v.std() / v.mean()) if v.size and v.mean() > 0 else 0.0

        return {
            "length": float(len(regions)),
            "area_mean": float(areas.mean()) if areas.size else 0.0,
            "area_cv": cv(areas),
            "area_growth": float((areas[-1] - areas[0]) / areas[0]) if areas.size and areas[0] > 0 else 0.0,
            "perimeter_cv": cv(perims),
            "circularity_std": float(circ.std()) if circ.size else 0.0,
            "centroid_shift": float(np.mean([center_distance(a.centroid, b.centroid) for a, b in steps])) if steps else 0.0,
            "bbox_iou_mean": float(np.mean([bbox_iou(a.bbox, b.bbox) for a, b in steps])) if steps else 1.0,
        }


class RegionTracker:
    """Gán track_id cho Region theo từng frame.

    min_iou / max_center_distance: ngưỡng ghép (mặc định lấy từ Config).
    history: số frame lịch sử giữ lại mỗi track.
    max_missed: số frame liên tiếp được phép mất dấu trước khi xóa track.
    """

    def __init__(self, cfg=None, min_iou: float | None = None, max_center_distance: float | None = None,
                 history: int | None = None, max_missed: int = 2):
        self.min_iou = min_iou if min_iou is not None else getattr(cfg, "min_iou", 0.1)
        self.max_center_distance = (max_center_distance if max_center_distance is not None
                                    else getattr(cfg, "max_center_distance", 55.0))
        self.history = history if history is not None else getattr(cfg, "vote_window", 7)
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next_id = 1

    def _match_cost(self, track: Track, region) -> float | None:
        """Chi phí ghép nhỏ hơn là tốt hơn; None nếu không được ghép."""
        if track.kind != region.kind:
            return None
        iou = bbox_iou(track.last.bbox, region.bbox)
        if iou >= self.min_iou:
            return 1.0 - iou  # trong [0, 1)
        dist = center_distance(track.last.centroid, region.centroid)
        if dist <= self.max_center_distance:
            return 1.0 + dist / self.max_center_distance  # luôn kém hơn ghép bằng IoU
        return None

    def update(self, regions: list, frame_idx: int) -> list[tuple[int, object]]:
        """Trả về danh sách (track_id, region) theo đúng thứ tự `regions`."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for ri, region in enumerate(regions):
                cost = self._match_cost(track, region)
                if cost is not None:
                    pairs.append((cost, ti, ri))
        pairs.sort()
        used_t, assigned = set(), {}
        for _, ti, ri in pairs:
            if ti in used_t or ri in assigned:
                continue
            used_t.add(ti)
            assigned[ri] = self.tracks[ti]

        for ri, region in enumerate(regions):
            track = assigned.get(ri)
            if track is None:
                track = Track(self._next_id, region.kind, deque(maxlen=self.history))
                self._next_id += 1
                self.tracks.append(track)
                assigned[ri] = track
            track.history.append((frame_idx, region))
            track.missed = 0

        matched = {id(t) for t in assigned.values()}
        for track in self.tracks:
            if id(track) not in matched:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [(assigned[ri].track_id, region) for ri, region in enumerate(regions)]

    def get(self, track_id: int) -> Track | None:
        return next((t for t in self.tracks if t.track_id == track_id), None)
