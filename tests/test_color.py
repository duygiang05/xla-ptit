import numpy as np
import pytest

from fireguard.color import color_candidates
from fireguard.color.preprocessing import (denoise, preprocess_color, resize_keep_aspect,
                                           to_gray, to_ycrcb)
from fireguard.config import Config


def test_color_masks_shape_and_type():
    frame=np.zeros((40,60,3),dtype=np.uint8)
    masks=color_candidates(frame,Config())
    assert set(masks)=={'fire','smoke'}
    for m in masks.values():
        assert m.shape==(40,60)
        assert m.dtype==np.uint8
        assert set(np.unique(m)).issubset({0,255})


# ------------------------------------------------------------ tiền xử lý
def test_resize_keep_aspect_landscape_and_portrait():
    # 1080p ngang -> vừa khít 640x360; video dọc giữ tỉ lệ 9:16 (không bị kéo méo)
    wide = np.zeros((1080, 1920, 3), np.uint8)
    assert resize_keep_aspect(wide, 640, 360).shape == (360, 640, 3)
    tall = np.zeros((1920, 1080, 3), np.uint8)
    h, w = resize_keep_aspect(tall, 640, 360).shape[:2]
    assert h == 360 and abs(w / h - 1080 / 1920) < 0.01


def test_resize_never_upscales():
    small = np.zeros((100, 200, 3), np.uint8)
    assert resize_keep_aspect(small, 640, 360).shape == small.shape


def test_denoise_off_and_even_kernel():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    # kernel <= 1: tắt blur, ảnh giữ nguyên
    assert np.array_equal(denoise(noisy, 1), noisy)
    # kernel chẵn không làm OpenCV báo lỗi, và blur làm giảm dao động nhiễu
    out = denoise(noisy, 4)
    assert out.shape == noisy.shape and out.std() < noisy.std()


def test_preprocess_color_hsv_of_orange():
    # Màu cam lửa trong BGR (B=0, G=128, R=255) -> Hue OpenCV ~ 15 (thang 0–179)
    frame = np.full((20, 20, 3), (0, 128, 255), np.uint8)
    hsv = preprocess_color(frame, Config())
    assert hsv.shape == frame.shape and hsv.dtype == np.uint8
    h, s, v = hsv[10, 10]
    assert 10 <= h <= 20 and s == 255 and v == 255


def test_other_color_spaces_shapes():
    frame = np.zeros((30, 40, 3), np.uint8)
    assert to_ycrcb(frame, Config()).shape == (30, 40, 3)
    assert to_gray(frame, Config()).shape == (30, 40)


@pytest.mark.parametrize("bad", [np.zeros((10, 10), np.uint8),          # ảnh xám
                                 np.zeros((10, 10, 3), np.float32)])    # sai dtype
def test_preprocess_rejects_bad_frames(bad):
    with pytest.raises(ValueError):
        preprocess_color(bad, Config())


# ------------------------------------------------------------ mask lửa
from dataclasses import replace

from fireguard.color.fire_detection import fire_mask_hsv, fire_mask_rgb


def _fire_ratio(bgr, **cfg_changes):
    """Tỉ lệ pixel được đánh dấu lửa trên ảnh 20x20 tô một màu đồng nhất."""
    frame = np.full((20, 20, 3), bgr, np.uint8)
    cfg = replace(Config(), blur_kernel=1, **cfg_changes)
    return np.count_nonzero(color_candidates(frame, cfg)['fire']) / frame[:, :, 0].size


@pytest.mark.parametrize("bgr, expected", [
    ((0, 128, 255), 1.0),    # cam lửa
    ((0, 220, 255), 1.0),    # vàng lửa hue = 26 (ngưỡng cũ 25 bỏ sót)
    ((60, 0, 220), 1.0),     # đỏ thẫm hue ~172: chỉ bắt được nhờ khoảng quấn 170–179
    ((0, 40, 80), 0.0),      # cam tối (V thấp)
    ((128, 128, 128), 0.0),  # tường xám (S = 0)
    ((255, 0, 0), 0.0),      # xanh lam
    ((0, 200, 0), 0.0),      # xanh lá
])
def test_fire_hsv_mode(bgr, expected):
    assert _fire_ratio(bgr) == expected


def test_wrap_hue_can_be_disabled():
    assert _fire_ratio((60, 0, 220), fire_h_wrap_low=180) == 0.0


def test_rgb_rule_rejects_pinkish_red_but_keeps_orange():
    pink_red = (90, 40, 200)   # áo đỏ ngả hồng: B > G
    assert _fire_ratio(pink_red, fire_color_mode="hsv") == 1.0   # HSV không phân biệt được
    assert _fire_ratio(pink_red, fire_color_mode="rgb") == 0.0
    assert _fire_ratio(pink_red, fire_color_mode="and") == 0.0   # kết hợp -> loại được
    assert _fire_ratio(pink_red, fire_color_mode="or") == 1.0
    for mode in ("rgb", "and", "or"):
        assert _fire_ratio((0, 128, 255), fire_color_mode=mode) == 1.0


def test_fire_rules_output_format():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    hsv = preprocess_color(frame, Config())
    for m in (fire_mask_hsv(hsv, Config()), fire_mask_rgb(frame, Config())):
        assert m.shape == (30, 40) and m.dtype == np.uint8
        assert set(np.unique(m)).issubset({0, 255})


def test_invalid_fire_mode_raises():
    with pytest.raises(ValueError):
        _fire_ratio((0, 128, 255), fire_color_mode="abc")
