import cv2
import numpy as np
from fireguard.config import Config
from .preprocessing import preprocess_color
from .smoke_detection import smoke_candidates

def fire_candidates(hsv, cfg: Config):
    lo = np.array([cfg.fire_h_low, cfg.fire_s_min, cfg.fire_v_min], dtype=np.uint8)
    hi = np.array([cfg.fire_h_high, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)

def color_candidates(frame_bgr, cfg: Config):
    """TV1 baseline: masks 0/255, only candidates—not validated detections."""
    hsv = preprocess_color(frame_bgr, cfg)
    return {'fire': fire_candidates(hsv, cfg), 'smoke': smoke_candidates(hsv, cfg)}
