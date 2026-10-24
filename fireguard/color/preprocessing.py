import cv2
from fireguard.config import Config

def preprocess_color(frame_bgr, cfg: Config):
    blurred = cv2.GaussianBlur(frame_bgr, (cfg.blur_kernel, cfg.blur_kernel), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
