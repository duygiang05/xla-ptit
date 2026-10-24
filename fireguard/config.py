from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Placeholder defaults: tune on development videos, not on held-out test.
    width: int = 640
    height: int = 360
    blur_kernel: int = 5
    fire_h_low: int = 0
    fire_h_high: int = 25
    fire_s_min: int = 100
    fire_v_min: int = 125
    smoke_s_max: int = 70
    smoke_v_min: int = 65
    smoke_v_max: int = 230
    motion_threshold: int = 18
    min_region_area: int = 130
    morph_kernel: int = 3
    vote_window: int = 7
    vote_hits: int = 3
    min_iou: float = 0.1
    max_center_distance: float = 55.0
