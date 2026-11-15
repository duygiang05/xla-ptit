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
