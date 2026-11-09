import numpy as np

from fireguard.config import Config
from fireguard.motion import MotionDetector
from fireguard.motion.background import MOG2MotionDetector


# Test 1: Frame dau tien va frame co chuyen dong
def test_first_frame_zero_and_second_frame_motion():
    d = MotionDetector(Config())

    base = np.zeros((40, 60, 3), dtype=np.uint8)

    assert not d.apply(base).any()

    moved = base.copy()
    moved[10:25, 10:25] = 255

    assert d.apply(moved).any()


# Test 2: Kiem tra dinh dang mask Frame Difference
def test_frame_difference_mask_format():
    detector = MotionDetector(Config())

    frame1 = np.zeros((100, 150, 3), dtype=np.uint8)

    frame2 = frame1.copy()
    frame2[30:70, 40:90] = 255

    detector.apply(frame1)
    mask = detector.apply(frame2)

    assert mask.shape == (100, 150)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


# Test 3: Khong co chuyen dong
def test_frame_difference_no_motion():
    detector = MotionDetector(Config())

    frame = np.zeros((100, 150, 3), dtype=np.uint8)

    detector.apply(frame)
    mask = detector.apply(frame.copy())

    assert np.count_nonzero(mask) == 0


# Test 4: Kiem tra dinh dang mask MOG2
def test_mog2_mask_format():
    detector = MOG2MotionDetector(
        history=200,
        var_threshold=16
    )

    frame = np.zeros((100, 150, 3), dtype=np.uint8)

    mask = detector.apply(frame)

    assert mask.shape == (100, 150)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
