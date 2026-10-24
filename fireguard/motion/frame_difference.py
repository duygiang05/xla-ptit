import cv2
import numpy as np
from fireguard.config import Config

class MotionDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._previous = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._previous is None:
            self._previous = gray
            return np.zeros_like(gray)
        diff = cv2.absdiff(gray, self._previous)
        _, mask = cv2.threshold(diff, self.cfg.motion_threshold, 255, cv2.THRESH_BINARY)
        self._previous = gray
        return mask
