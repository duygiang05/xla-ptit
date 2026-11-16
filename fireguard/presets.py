"""Cấu hình OpenCV cho giao diện và CLI."""

DEMO_OPTIONS = {
    "fire_fusion": "and_dilate",
    "smoke_fusion": "region_motion",
    "motion_dilate": 5,
    "smoke_motion": "mog2",
    "smoke_motion_dilate": 9,
    "smoke_temporal_mode": "vote",
    "smoke_verify": "shape_edge",
    "smoke_max_aspect_ratio": 1.10,
    "smoke_max_edge_density": 0.02,
    "single_box_per_kind": True,
}

# Giao diện chỉ dùng một chế độ khói theo bối cảnh.
OPENCV_CONTEXT_OPTIONS = {
    "smoke_engine": "opencv_context",
    "smoke_motion": "mog2",
    "smoke_fusion": "region_motion",
    "smoke_motion_dilate": 9,
    "smoke_verify": "off",
    "smoke_vote_hits": 5,
}
