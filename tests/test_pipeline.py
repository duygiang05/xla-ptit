from pathlib import Path
import numpy as np
import cv2
import pytest
import fireguard.pipeline as pipeline
from fireguard.pipeline import run_video
from fireguard.config import Config
from fireguard.color import color_candidates
from fireguard.shape import extract_regions


def test_pipeline_generates_video(tmp_path: Path):
    p = tmp_path / "in.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    assert w.isOpened()
    for i in range(8):
        f = np.zeros((90, 160, 3), dtype=np.uint8)
        cv2.rectangle(f, (10 + i * 5, 10), (50 + i * 5, 70), (0, 100, 255), -1)
        w.write(f)
    w.release()
    out = tmp_path / "out.mp4"
    dashboard = tmp_path / "dashboard.mp4"
    csv_path = tmp_path / "out.csv"
    result = run_video(
        str(p),
        str(out),
        str(csv_path),
        Config(width=160, height=90, min_region_area=20),
        dashboard_path=str(dashboard),
    )
    assert result["frames"] == 8
    assert out.exists() and out.stat().st_size > 0 and csv_path.exists()
    assert dashboard.exists() and dashboard.stat().st_size > 0
    output_cap = cv2.VideoCapture(str(out))
    dashboard_cap = cv2.VideoCapture(str(dashboard))
    try:
        assert int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        assert int(dashboard_cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
        assert output_cap.get(cv2.CAP_PROP_FPS) == dashboard_cap.get(cv2.CAP_PROP_FPS)
        assert (int(dashboard_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(dashboard_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (720, 648)
        for _ in range(8):
            output_ok, output_frame = output_cap.read()
            dashboard_ok, dashboard_frame = dashboard_cap.read()
            assert output_ok and dashboard_ok
            assert output_frame.shape == (90, 160, 3)
            assert dashboard_frame.shape == (648, 720, 3)
    finally:
        output_cap.release()
        dashboard_cap.release()


def test_same_frame_can_hold_fire_and_smoke_candidates():
    # Kiểm thử giao tiếp module đa nhãn; KHÔNG khẳng định thuật toán phát hiện ảnh thực chính xác.
    cfg = Config(width=160, height=90, min_region_area=15)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (55, 70), (0, 100, 255), -1)  # vùng màu lửa
    cv2.rectangle(frame, (100, 10), (145, 70), (130, 130, 130), -1)  # vùng màu khói
    candidates = color_candidates(frame, cfg)
    assert set(candidates) == {"fire", "smoke"}
    assert candidates["fire"].shape == candidates["smoke"].shape == frame.shape[:2]
    assert np.count_nonzero(candidates["fire"]) > 0
    assert np.count_nonzero(candidates["smoke"]) > 0
    regions = extract_regions(candidates["fire"], "fire", cfg) + extract_regions(
        candidates["smoke"], "smoke", cfg
    )
    assert {"fire", "smoke"} <= {r.kind for r in regions}


@pytest.mark.parametrize("selected,expected", [
    (None, "adaptive_region_motion"),
    ("region_motion", "region_motion"),
    ("adaptive_region_motion", "adaptive_region_motion"),
])
def test_context_ordinary_scene_uses_selected_smoke_fusion(tmp_path, monkeypatch, selected, expected):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    writer.write(np.full((48, 64, 3), 120, np.uint8))
    writer.release()
    monkeypatch.setattr(pipeline, "active_image_bounds", lambda frame: (0, 0, 64, 48))
    monkeypatch.setattr(
        pipeline, "context_smoke_candidates",
        lambda frame, fire, **kwargs: (np.full((48, 64), 255, np.uint8), False, False, None),
    )
    seen = []
    original = pipeline.fuse_masks

    def recording_fuse(color, motion, **kwargs):
        seen.append(kwargs["mode"])
        return original(color, motion, **kwargs)

    monkeypatch.setattr(pipeline, "fuse_masks", recording_fuse)
    options = {} if selected is None else {"smoke_fusion": selected}
    pipeline.run_video(
        str(source), str(tmp_path / "result.mp4"),
        cfg=Config(width=64, height=48), smoke_engine="opencv_context", **options,
    )
    assert seen == ["and", expected]


@pytest.mark.parametrize("sky,aerial,portrait,expected,dilate", [
    (True, False, False, "sky_motion", 41),
    (False, True, False, "and_dilate", 21),
    (False, False, True, "and_dilate", 13),
])
def test_context_special_scenes_keep_fusion_modes(
    tmp_path, monkeypatch, sky, aerial, portrait, expected, dilate
):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    writer.write(np.full((48, 64, 3), 120, np.uint8))
    writer.release()
    bounds = (20, 0, 20, 48) if portrait else (0, 0, 64, 48)
    monkeypatch.setattr(pipeline, "active_image_bounds", lambda frame: bounds)
    monkeypatch.setattr(
        pipeline, "context_smoke_candidates",
        lambda frame, fire, **kwargs: (
            np.full((48, 64), 255, np.uint8), sky, aerial, np.zeros((48, 64), np.uint8)
        ),
    )
    seen = []
    original = pipeline.fuse_masks

    def recording_fuse(color, motion, **kwargs):
        seen.append((kwargs["mode"], kwargs["motion_dilate"]))
        return original(color, motion, **kwargs)

    monkeypatch.setattr(pipeline, "fuse_masks", recording_fuse)
    pipeline.run_video(
        str(source), str(tmp_path / "result.mp4"),
        cfg=Config(width=64, height=48), smoke_engine="opencv_context",
        smoke_fusion="adaptive_region_motion",
    )
    assert seen[1] == (expected, dilate)
