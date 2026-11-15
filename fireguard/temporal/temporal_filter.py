"""TV4: xác nhận vùng nghi lửa/khói bằng lịch sử TV3 RegionTracker.

Pipeline không cần sửa: TemporalFilter(cfg).update(regions, frame_idx) -> list[Region].
Mặc định là vote-only (baseline); mode='hybrid' bổ sung lọc biến dạng hình học.
Không tự gán điểm tin cậy xác suất vào Region.score.
"""
from collections import deque
from fireguard.config import Config
from fireguard.shape.regions import Region
from fireguard.shape.tracking import RegionTracker, Track
from .decision import ShapeThresholds, passes_temporal_vote, passes_shape_evidence


class TemporalFilter:
    """Nhận các Region của TV3, trả các Region hiện tại được xác nhận.

    - Các track dùng CHÍNH RegionTracker của TV3, không tự triển khai lần hai.
    - K/N đếm frame có quan sát thật (kể cả frame hiện tại), không đếm lần mất dấu.
    - Mặc định vote bảo toàn hành vi pipeline; mode='hybrid' thêm shape evidence.
    - last_decisions: dữ liệu debug từng Region, không phải ground truth.
    """

    def __init__(self, cfg: Config, decision_mode: str = 'vote',
                 shape_thresholds: ShapeThresholds | None = None,
                 max_missed: int = 2):
        if cfg.vote_window < 1 or not 1 <= cfg.vote_hits <= cfg.vote_window:
            raise ValueError('Cần 1 <= vote_hits <= vote_window')
        if decision_mode not in ('vote', 'hybrid'):
            raise ValueError("decision_mode phải là 'vote' hoặc 'hybrid'")
        if max_missed < 0:
            raise ValueError('max_missed không được âm')
        self.cfg = cfg
        self.decision_mode = decision_mode
        self.shape_thresholds = shape_thresholds or ShapeThresholds()
        self.max_missed = max_missed
        self.tracker = RegionTracker(cfg, history=cfg.vote_window, max_missed=max_missed)
        self._last_frame_idx = -1
        self.last_decisions: list[dict] = []

    def update(self, regions: list[Region], frame_idx: int) -> list[Region]:
        """Gọi một lần/frame; truyền [] cho frame không có ứng viên.

        Chấp nhận frame_idx tăng không liên tiếp: mô phỏng frame trống bị bỏ qua.
        Một track có thể mất tối đa max_missed frame, giống TV3 tracker.
        """
        if frame_idx <= self._last_frame_idx:
            raise ValueError('frame_idx phải tăng theo thời gian')
        if self._last_frame_idx >= 0:
            gap = frame_idx - self._last_frame_idx - 1
            if gap > self.max_missed:
                # Không lặp qua hàng ngàn frame bị bỏ; mọi track đã hết hạn.
                self.tracker.tracks.clear()
            else:
                for missed_frame in range(self._last_frame_idx + 1, frame_idx):
                    self.tracker.update([], missed_frame)
        self._last_frame_idx = frame_idx
        self.last_decisions = []
        assignments = self.tracker.update(regions, frame_idx)
        result = []
        earliest = frame_idx - self.cfg.vote_window + 1
        for track_id, region in assignments:
            track = self.tracker.get(track_id)
            assert track is not None
            # TV3 cung cấp history [(frame_idx, Region), ...].
            hits = len({idx for idx, _ in track.history if earliest <= idx <= frame_idx})
            # Track.history giữ N QUAN SÁT; lọc thêm theo N FRAME thực tế.
            # Tạo snapshot Track từ TV3 để tái sử dụng chính Track.shape_change().
            recent = deque(((idx, item) for idx, item in track.history
                            if earliest <= idx <= frame_idx),
                           maxlen=self.cfg.vote_window)
            shape = Track(track.track_id, track.kind, recent).shape_change()
            vote_ok = passes_temporal_vote(hits, self.cfg.vote_hits)
            shape_ok = passes_shape_evidence(region.kind, shape, self.shape_thresholds)
            accepted = vote_ok and (self.decision_mode == 'vote' or shape_ok)
            if accepted:
                result.append(region)
            self.last_decisions.append({
                'frame': frame_idx, 'track_id': track_id, 'kind': region.kind,
                'bbox': region.bbox, 'hits': hits, 'window': self.cfg.vote_window,
                'vote_pass': vote_ok, 'shape_pass': shape_ok,
                'confirmed': accepted, 'mode': self.decision_mode,
                **shape,
            })
        return result
