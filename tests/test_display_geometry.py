import numpy as np

from fireguard.pipeline import letterbox_frame
from fireguard.shape.grouping import union_regions
from fireguard.shape.regions import Region


def region(kind, x, y, w, h):
    return Region(kind, (x, y, w, h), w * h, (x + w / 2, y + h / 2))


def test_letterbox_preserves_portrait_and_landscape():
    portrait = np.full((960, 540, 3), 200, dtype=np.uint8)
    result = letterbox_frame(portrait, 640, 360)
    assert result.shape == (360, 640, 3)
    assert np.all(result[:, :200] == 0)
    assert np.all(result[:, 220:420] == 200)
    landscape = np.full((240, 320, 3), 150, dtype=np.uint8)
    result = letterbox_frame(landscape, 640, 360)
    assert result.shape == (360, 640, 3)
    assert np.all(result[:, 80:560] == 150)


def test_union_region_keeps_one_box_per_kind():
    pieces = [
        region("fire", 10, 20, 20, 30),
        region("fire", 42, 25, 25, 30),
        region("fire", 250, 200, 10, 10),
        region("smoke", 5, 5, 20, 20),
    ]
    assert union_regions(pieces, "smoke")[0].bbox == (5, 5, 20, 20)
    assert union_regions(pieces, "fire")[0].bbox == (10, 20, 250, 190)
