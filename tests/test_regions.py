import math

import cv2
import numpy as np
import pytest

from fireguard.config import Config
from fireguard.shape import (Region, RegionTracker, apply_morphology, bbox_iou, clean_mask,
                             combine_masks, draw_regions, extract_regions, extract_regions_cc,
                             shape_features)


def test_regions_and_iou():
    mask=np.zeros((90,90),dtype=np.uint8)
    cv2.rectangle(mask,(10,10),(35,35),255,-1)
    regions=extract_regions(mask,'fire',Config())
    assert len(regions)==1
    assert regions[0].kind=='fire'
    assert bbox_iou((0,0,10,10),(0,0,10,10))==1


# ---------- Region schema ----------

def test_region_backward_compatible_constructor():
    r = Region('fire', (5, 5, 20, 20), 400, (15, 15))
    assert r.score == 1.0 and r.perimeter == 0.0


def test_region_has_bbox_area_perimeter_centroid():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 30), (59, 49), 255, -1)  # 40x20
    # clean=False: opening với kernel ellipse sẽ bo tròn 4 góc, làm lệch chu vi
    (r,) = extract_regions(mask, 'smoke', Config(min_region_area=10), clean=False)
    assert r.kind == 'smoke'
    assert r.bbox == (20, 30, 40, 20)
    assert r.area == pytest.approx(39 * 19, rel=0.02)  # contourArea đo theo đa giác biên
    assert r.perimeter == pytest.approx(2 * (39 + 19), rel=0.02)
    assert r.centroid == pytest.approx((39.5, 39.5), abs=0.6)
    assert r.aspect_ratio == pytest.approx(2.0)


# ---------- Shape features ----------

def _contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def test_circle_is_round_and_solid():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    f = shape_features(_contour(mask))
    assert f.circularity > 0.85
    assert f.solidity > 0.98
    assert f.extent == pytest.approx(math.pi / 4, abs=0.03)


def test_jagged_flame_like_shape_has_low_circularity_and_solidity():
    mask = np.zeros((200, 200), dtype=np.uint8)
    pts = []
    for i in range(16):  # hình sao nhiều "lưỡi lửa"
        ang = 2 * math.pi * i / 16
        rad = 80 if i % 2 == 0 else 30
        pts.append((100 + rad * math.cos(ang), 100 + rad * math.sin(ang)))
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    f = shape_features(_contour(mask))
    assert f.circularity < 0.5
    assert f.solidity < 0.7


# ---------- Morphology ----------

def test_opening_removes_isolated_noise():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)
    # nhiễu 2x2 pixel (1 pixel đơn lẻ có contourArea = 0 nên đã bị lọc sẵn)
    for y, x in [(5, 5), (90, 10), (50, 95)]:
        mask[y:y + 2, x:x + 2] = 255
    cfg = Config(min_region_area=1)
    assert len(extract_regions(mask, 'fire', cfg, clean=False)) == 4
    assert len(extract_regions(mask, 'fire', cfg, clean=True)) == 1


def test_closing_fills_small_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    mask[50, 50] = mask[40, 60] = 0
    closed = apply_morphology(mask, ('close',), ksize=3)
    assert closed[50, 50] == 255 and closed[40, 60] == 255


def test_clean_mask_keeps_binary_format():
    rng = np.random.default_rng(0)
    mask = (rng.random((60, 80)) > 0.7).astype(np.uint8) * 255
    out = clean_mask(mask, Config())
    assert out.shape == mask.shape and out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_combine_masks_modes():
    color = np.zeros((20, 20), dtype=np.uint8); color[:, :10] = 255
    motion = np.zeros((20, 20), dtype=np.uint8); motion[:, 8:12] = 255
    assert np.count_nonzero(combine_masks(color, motion, 'and')) == 20 * 2
    assert np.count_nonzero(combine_masks(color, motion, 'or')) == 20 * 12
    assert np.array_equal(combine_masks(color, motion, 'color'), color)
    assert np.count_nonzero(combine_masks(color, motion, 'and', motion_dilate=5)) > 20 * 2
    with pytest.raises(ValueError):
        combine_masks(color, motion, 'xor')


# ---------- connectedComponentsWithStats vs findContours ----------

def test_connected_components_matches_contours():
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (40, 50), 255, -1)
    cv2.circle(mask, (110, 70), 25, 255, -1)
    cfg = Config(min_region_area=20)
    a = sorted(extract_regions(mask, 'fire', cfg), key=lambda r: r.bbox)
    b = sorted(extract_regions_cc(mask, 'fire', cfg), key=lambda r: r.bbox)
    assert len(a) == len(b) == 2
    for ra, rb in zip(a, b):
        assert bbox_iou(ra.bbox, rb.bbox) > 0.9
        assert rb.area == pytest.approx(ra.area, rel=0.15)  # pixel count vs polygon area
        assert rb.centroid == pytest.approx(ra.centroid, abs=1.0)


def test_min_region_area_filters_small_components():
    mask = np.zeros((60, 60), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (9, 9), 255, -1)    # 25 px
    cv2.rectangle(mask, (20, 20), (49, 49), 255, -1)
    cfg = Config(min_region_area=100)
    assert len(extract_regions(mask, 'fire', cfg)) == 1
    assert len(extract_regions_cc(mask, 'fire', cfg)) == 1


def test_draw_regions_does_not_modify_input():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    out = draw_regions(frame, [Region('fire', (10, 10, 20, 20), 400, (20, 20))])
    assert not frame.any() and out.any()


# ---------- Tracker ----------

def _r(kind, x, y, w=20, h=20):
    return Region(kind, (x, y, w, h), float(w * h), (x + w / 2, y + h / 2), 1.0, 2.0 * (w + h), 0.7)


def test_tracker_keeps_id_for_moving_region():
    tr = RegionTracker(Config())
    ids = [tr.update([_r('fire', 10 + 4 * i, 10)], i)[0][0] for i in range(5)]
    assert len(set(ids)) == 1
    assert tr.get(ids[0]).length == 5


def test_tracker_separates_kinds_and_far_regions():
    tr = RegionTracker(Config())
    out = tr.update([_r('fire', 10, 10), _r('smoke', 10, 10), _r('fire', 200, 200)], 0)
    assert len({tid for tid, _ in out}) == 3
    out2 = tr.update([_r('smoke', 12, 10), _r('fire', 202, 200), _r('fire', 11, 10)], 1)
    assert [tid for tid, _ in out2] == [out[1][0], out[2][0], out[0][0]]


def test_tracker_drops_lost_tracks_after_max_missed():
    tr = RegionTracker(Config(), max_missed=1)
    tid = tr.update([_r('fire', 10, 10)], 0)[0][0]
    tr.update([], 1)
    assert tr.get(tid) is not None
    tr.update([], 2)
    assert tr.get(tid) is None
    assert tr.update([_r('fire', 10, 10)], 3)[0][0] != tid


def test_shape_change_static_vs_flickering():
    static = RegionTracker(Config())
    flicker = RegionTracker(Config())
    for i in range(6):
        sid = static.update([_r('fire', 50, 50)], i)[0][0]
        w = 20 + (8 if i % 2 else -6)
        fid = flicker.update([_r('fire', 50, 50, w, w + 4)], i)[0][0]
    s = static.get(sid).shape_change()
    f = flicker.get(fid).shape_change()
    assert s['area_cv'] == pytest.approx(0.0) and s['centroid_shift'] == pytest.approx(0.0)
    assert f['area_cv'] > 0.2
    assert s['length'] == f['length'] == 6
