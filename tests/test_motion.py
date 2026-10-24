import numpy as np
from fireguard.motion import MotionDetector
from fireguard.config import Config

def test_first_frame_zero_and_second_frame_motion():
    d=MotionDetector(Config())
    base=np.zeros((40,60,3),dtype=np.uint8)
    assert not d.apply(base).any()
    moved=base.copy();moved[10:25,10:25]=255
    assert d.apply(moved).any()
