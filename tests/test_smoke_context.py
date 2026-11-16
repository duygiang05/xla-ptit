import cv2
import numpy as np

from fireguard.color.smoke_context import context_smoke_candidates
from fireguard.config import Config
from fireguard import pipeline
from fireguard.shape.context_verify import (
    active_image_bounds,
    verify_context_regions,
    extend_sky_plume,
)
from fireguard.shape.regions import Region
from fireguard.motion.flow import moving_fraction


def test_context_candidates_are_binary_and_report_scene():
    frame = np.full((90, 160, 3), (70, 110, 170), dtype=np.uint8)
    frame[10:70, 20:100] = (150, 150, 150)
    fire = np.zeros((90, 160), dtype=np.uint8)
    mask, sky = context_smoke_candidates(frame, fire)
    assert sky is False
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_context_verification_rejects_full_frame_wall():
    frame = np.full((90, 160, 3), 150, dtype=np.uint8)
    mask = np.full((90, 160), 255, dtype=np.uint8)
    region = Region("smoke", (0, 0, 160, 90), 14400, (80, 45))
    assert verify_context_regions([region], frame, mask, False) == []


def test_context_keeps_large_upper_diffuse_plume_but_not_full_height_wall():
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=np.uint8)
    mask[:290, 20:640:2] = 255
    movement = np.zeros_like(mask)
    movement[20:100, 30:70] = 255
    plume = Region("smoke", (20, 0, 620, 290), 50000, (330, 145))
    wall = Region("smoke", (20, 0, 620, 360), 190000, (330, 180))
    assert verify_context_regions(
        [plume], frame, mask, False, motion_mask=movement
    ) == [plume]
    assert (
        verify_context_regions([wall], frame, mask, False, motion_mask=movement) == []
    )


def test_extend_portrait_sky_plume_with_low_edge_lower_column():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, 219:421] = 120
    upper = Region("smoke", (240, 50, 120, 120), 10000, (300, 110))
    lower = Region("smoke", (225, 170, 190, 130), 10000, (320, 235))
    joined = extend_sky_plume([upper], [lower], frame)
    assert len(joined) == 1
    assert joined[0].bbox == (225, 50, 190, 250)


def test_aerial_tint_does_not_enable_color_only_fusion():
    hsv = np.full((90, 160, 3), (100, 65, 160), dtype=np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fire = np.zeros((90, 160), dtype=np.uint8)
    mask, sky, aerial = context_smoke_candidates(frame, fire, return_aerial_tint=True)
    assert mask.shape == (90, 160)
    assert not sky
    assert aerial


def test_strong_blue_sky_keeps_gray_smoke_and_excludes_sky():
    hsv = np.full((90, 160, 3), (103, 130, 165), dtype=np.uint8)
    hsv[10:80, 15:70] = (103, 45, 140)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fire = np.zeros((90, 160), dtype=np.uint8)
    mask, sky, aerial = context_smoke_candidates(frame, fire, return_aerial_tint=True)
    assert sky and not aerial
    assert mask[40, 40] == 255
    assert mask[40, 120] == 0


def test_moderately_saturated_blue_sky_does_not_merge_dim_sky_with_smoke():
    hsv = np.full((90, 160, 3), (103, 90, 120), dtype=np.uint8)
    hsv[:30, 95:] = (103, 90, 170)
    hsv[10:80, 15:70] = (103, 45, 140)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fire = np.zeros((90, 160), dtype=np.uint8)
    mask, sky, aerial = context_smoke_candidates(frame, fire, return_aerial_tint=True)
    assert sky and not aerial
    assert mask[40, 40] == 255
    assert mask[70, 120] == 0


def test_moderate_sky_reports_lower_candidate_without_exposing_it_to_color_only():
    hsv = np.full((360, 640, 3), (18, 75, 125), dtype=np.uint8)
    hsv[:50] = (103, 130, 165)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fire = np.zeros((360, 640), dtype=np.uint8)
    upper, sky, aerial, lower = context_smoke_candidates(
        frame, fire, return_aerial_tint=True, return_lower=True
    )
    assert sky and not aerial
    assert upper[250, 300] == 0
    assert lower[250, 300] == 255
    assert lower[20, 300] == 0


def test_aerial_tint_rejects_top_only_and_full_frame_regions():
    frame = np.full((90, 160, 3), 150, dtype=np.uint8)
    mask = np.full((90, 160), 255, dtype=np.uint8)
    top = Region("smoke", (20, 0, 50, 15), 700, (45, 7))
    wall = Region("smoke", (0, 0, 160, 90), 14400, (80, 45))
    plume = Region("smoke", (50, 35, 40, 50), 1800, (70, 60))
    assert verify_context_regions(
        [top, wall, plume], frame, mask, False, aerial_tint=True
    ) == [plume]


def test_portrait_smoke_geometry_uses_content_width_and_rejects_fire_overlap():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[:, 50:110] = 150
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[30:80, 55:105] = 255
    region = Region("smoke", (55, 30, 50, 50), 2500, (80, 55))
    flame = Region("fire", (60, 35, 40, 40), 1600, (80, 55))
    assert active_image_bounds(frame) == (50, 0, 60, 90)
    assert verify_context_regions([region], frame, mask, False) == [region]
    assert (
        verify_context_regions([region], frame, mask, False, fire_regions=[flame]) == []
    )


def test_dense_flow_region_fraction_uses_letterbox_offset():
    flow = np.zeros((40, 60, 2), dtype=np.float32)
    flow[10:30, 15:35, 0] = 1.0
    assert moving_fraction(flow, (65, 20, 20, 20), (50, 10, 60, 40)) == 1.0
    assert moving_fraction(flow, (90, 20, 20, 20), (50, 10, 60, 40)) == 0.0


def test_sky_plume_extension_follows_lower_column_and_ignores_building():
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    # A building has many hard vertical edges, unlike the smooth plume.
    for x in range(550, 640, 8):
        frame[200:360, x : x + 2] = 255
    upper = Region("smoke", (150, 0, 490, 205), 45000, (395, 102))
    plume = Region("smoke", (215, 205, 155, 155), 8000, (292, 282))
    building = Region("smoke", (550, 210, 90, 150), 8000, (595, 285))
    extended = extend_sky_plume([upper], [building, plume], frame)
    assert len(extended) == 1
    assert extended[0].bbox == (151, 0, 283, 360)
    assert extend_sky_plume([upper], [building], frame) == [upper]


def test_sky_plume_extension_joins_narrow_head_across_short_color_gap():
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    head = Region("smoke", (179, 2, 99, 105), 4186, (228, 54))
    lower = Region("smoke", (0, 155, 223, 84), 15026, (111, 197))
    extended = extend_sky_plume([head], [lower], frame)
    assert extended[0].bbox == (0, 2, 278, 237)
    other = Region("smoke", (174, 171, 302, 87), 8845, (325, 214))
    assert extend_sky_plume([head], [lower, other], frame)[0].bbox == (0, 2, 476, 256)


def test_context_pipeline_processes_video_without_model(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for shift in range(8):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[10:30, shift : shift + 20] = 120
        writer.write(frame)
    writer.release()
    result = pipeline.run_video(
        str(source),
        str(tmp_path / "result.mp4"),
        cfg=Config(width=64, height=48, min_region_area=10),
        smoke_engine="opencv_context",
        smoke_vote_hits=5,
    )
    assert result["frames"] == 8
