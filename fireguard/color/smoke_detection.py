import cv2
from fireguard.config import Config

def smoke_candidates(hsv, cfg: Config):
    """Grayscale-like candidate only; motion and temporal validation required."""
    return cv2.inRange(hsv, (0, 0, cfg.smoke_v_min), (179, cfg.smoke_s_max, cfg.smoke_v_max))
