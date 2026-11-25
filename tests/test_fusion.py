import cv2
import numpy as np
import pytest
from fireguard.fusion import fuse_masks
from fireguard.shape.morphology import combine_masks


def two_masks():
    color = np.zeros((30, 30), np.uint8)
    color[8:22, 8:22] = 255
    motion = np.zeros_like(color)
    motion[10:12, 10:12] = 255
    return color, motion


def test_and_matches_old_pipeline():
    color, motion = two_masks()
    result = fuse_masks(color, motion, mode="and")
    assert np.array_equal(result, cv2.bitwise_and(color, motion))


def test_and_dilate_matches_tv3_helper():
    color, motion = two_masks()
    result = fuse_masks(color, motion, mode="and_dilate", motion_dilate=5)
    expected = combine_masks(color, motion, mode="and", motion_dilate=5)
    assert np.array_equal(result, expected)
    assert np.count_nonzero(result) > np.count_nonzero(fuse_masks(color, motion))
    assert np.count_nonzero(result) <= np.count_nonzero(color)


def test_color_only_is_copy():
    color, motion = two_masks()
    out = fuse_masks(color, motion, mode="color_only")
    assert np.array_equal(color, out)
    assert not np.shares_memory(color, out)


def test_sky_motion_uses_motion_without_expanding_past_color():
    color = np.full((80, 120), 255, np.uint8)
    motion = np.zeros_like(color)
    motion[30:40, 25:35] = 255
    result = fuse_masks(color, motion, mode="sky_motion", motion_dilate=21)
    assert np.count_nonzero(result) > np.count_nonzero(motion)
    assert np.count_nonzero(result) < np.count_nonzero(color)
    assert not np.any(result[:, 80:])
    assert not np.any(fuse_masks(color, np.zeros_like(motion), mode="sky_motion"))


def test_empty_motion_and_modes():
    color, motion = two_masks()
    motion[:] = 0
    assert not np.any(fuse_masks(color, motion, mode="and"))
    assert not np.any(fuse_masks(color, motion, mode="and_dilate"))
    assert np.array_equal(fuse_masks(color, motion, mode="color_only"), color)


@pytest.mark.parametrize("mode", ["and", "and_dilate", "color_only"])
def test_mask_format(mode):
    color, motion = two_masks()
    mask = fuse_masks(color, motion, mode=mode)
    assert mask.shape == color.shape and mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_bad_mode():
    color, motion = two_masks()
    with pytest.raises(ValueError):
        fuse_masks(color, motion, mode="or")


def test_bad_shape():
    color, motion = two_masks()
    with pytest.raises(ValueError):
        fuse_masks(color, motion[:20, :], mode="and")


def test_non_binary_mask():
    color, motion = two_masks()
    color[0, 0] = 17
    with pytest.raises(ValueError):
        fuse_masks(color, motion)


def test_invalid_kernel():
    color, motion = two_masks()
    with pytest.raises(ValueError):
        fuse_masks(color, motion, motion_dilate=0)


def test_zero_dilate_does_not_create_motion():
    color, motion = two_masks()
    motion[:] = 0
    assert not np.any(fuse_masks(color, motion, mode="and_dilate", motion_dilate=5))


def test_adaptive_small_component_matches_region_motion_and_preserves_inputs():
    color = np.zeros((100, 100), np.uint8)
    color[10:50, 10:50] = 255
    motion = np.zeros_like(color)
    motion[20:35, 20:35] = 255
    before_color, before_motion = color.copy(), motion.copy()
    old = fuse_masks(color, motion, mode="region_motion", motion_dilate=1)
    new = fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1)
    assert np.array_equal(new, old)
    assert np.array_equal(new, color)
    assert new.shape == color.shape and new.dtype == np.uint8
    assert set(np.unique(new)) == {0, 255}
    assert np.array_equal(color, before_color)
    assert np.array_equal(motion, before_motion)


def test_adaptive_large_component_keeps_only_sufficient_intersection():
    color = np.zeros((100, 100), np.uint8)
    color[:, :] = 255
    motion = np.zeros_like(color)
    motion[10:22, 10:22] = 255  # 144 pixels, less than 1.5% of the color region.
    old = fuse_masks(color, motion, mode="region_motion", motion_dilate=1)
    new = fuse_masks(
        color, motion, mode="adaptive_region_motion", motion_dilate=1,
        growth_iterations=0,
    )
    assert not np.any(old)
    assert np.array_equal(new, motion)


def test_adaptive_default_grows_large_region_from_motion_seed():
    color = np.full((100, 100), 255, np.uint8)
    motion = np.zeros_like(color)
    motion[40:52, 40:52] = 255
    grown = fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1)
    assert np.count_nonzero(grown) > np.count_nonzero(motion)
    assert np.all(grown[motion > 0] == 255)
    assert not np.any(grown[75:, :])


def test_adaptive_large_component_rejects_empty_and_small_intersections():
    color = np.zeros((100, 100), np.uint8)
    color[:80, :] = 255
    motion = np.zeros_like(color)
    assert not np.any(fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1))
    motion[10:20, 10:20] = 255
    assert not np.any(fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1))


def test_adaptive_filters_each_intersection_component_independently():
    color = np.zeros((100, 100), np.uint8)
    color[:80, :] = 255
    motion = np.zeros_like(color)
    motion[10:22, 10:22] = 255
    motion[40:50, 40:50] = 255
    expected = np.zeros_like(color)
    expected[10:22, 10:22] = 255
    assert np.array_equal(
        fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1,
                   growth_iterations=0),
        expected,
    )


def test_adaptive_area_ratio_boundary_is_small_component_path():
    color = np.zeros((100, 100), np.uint8)
    color[:50, :] = 255  # Exactly Q = 0.50.
    motion = np.zeros_like(color)
    motion[10:22, 10:22] = 255
    old = fuse_masks(color, motion, mode="region_motion", motion_dilate=1)
    new = fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1)
    assert np.array_equal(new, old)
    assert np.array_equal(new, color)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01, float("nan"), float("inf"), True])
def test_adaptive_rejects_invalid_area_ratio(ratio):
    color, motion = two_masks()
    with pytest.raises(ValueError, match="region_max_area_ratio"):
        fuse_masks(color, motion, mode="adaptive_region_motion", region_max_area_ratio=ratio)


def test_adaptive_area_ratio_is_configurable():
    color = np.zeros((100, 100), np.uint8)
    color[:40, :] = 255
    motion = np.zeros_like(color)
    motion[10:22, 10:22] = 255
    new = fuse_masks(
        color, motion, mode="adaptive_region_motion", motion_dilate=1,
        region_max_area_ratio=0.3, growth_iterations=0,
    )
    assert np.array_equal(new, motion)


def test_adaptive_handles_small_and_large_color_components_together():
    color = np.zeros((100, 100), np.uint8)
    color[:60, :90] = 255
    color[80:100, 10:30] = 255
    motion = np.zeros_like(color)
    motion[10:22, 10:22] = 255
    motion[85:90, 15:20] = 255
    expected = np.zeros_like(color)
    expected[10:22, 10:22] = 255
    expected[80:100, 10:30] = 255
    assert np.array_equal(
        fuse_masks(color, motion, mode="adaptive_region_motion", motion_dilate=1,
                   growth_iterations=0),
        expected,
    )
