import cv2


class MOG2MotionDetector:
    """Tách nền MOG2 để lấy chuyển động cho nhánh khói của app."""

    def __init__(self, history=200, var_threshold=16):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )

    def apply(self, frame_bgr):
        raw = self.subtractor.apply(frame_bgr)

        return cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)[1]
