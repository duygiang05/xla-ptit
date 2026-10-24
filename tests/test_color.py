import numpy as np
from fireguard.color import color_candidates
from fireguard.config import Config

def test_color_masks_shape_and_type():
    frame=np.zeros((40,60,3),dtype=np.uint8)
    masks=color_candidates(frame,Config())
    assert set(masks)=={'fire','smoke'}
    for m in masks.values():
        assert m.shape==(40,60)
        assert m.dtype==np.uint8
        assert set(np.unique(m)).issubset({0,255})
