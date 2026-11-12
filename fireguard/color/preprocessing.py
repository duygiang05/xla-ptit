"""TV1 – Tiền xử lý frame trước khi phân tích màu.

Thứ tự xử lý mặc định:  frame BGR  ->  GaussianBlur  ->  HSV
- Resize giữ tỉ lệ (`resize_keep_aspect`) là tiện ích cho thực nghiệm/đọc video;
  `color_candidates` KHÔNG tự resize vì pipeline đã resize và mask phải cùng kích thước frame.
- Lưu ý OpenCV: thứ tự kênh là BGR; Hue trong HSV là 0–179 (không phải 0–360).
"""
import cv2
import numpy as np

from fireguard.config import Config


def check_frame(frame_bgr: np.ndarray) -> None:
    """Báo lỗi sớm, dễ hiểu nếu frame sai định dạng (thay vì lỗi khó đọc của OpenCV)."""
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError("frame_bgr phải là np.ndarray")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"frame_bgr phải có shape (H,W,3), nhận {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"frame_bgr phải là uint8, nhận {frame_bgr.dtype}")


def resize_keep_aspect(frame_bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Thu nhỏ frame để vừa khung max_width×max_height mà KHÔNG méo hình.

    - Chỉ thu nhỏ, không phóng to (phóng to không thêm thông tin, chỉ tốn thời gian).
    - INTER_AREA là phép nội suy tốt nhất khi thu nhỏ: ít răng cưa, ít nhiễu moiré.
    Ví dụ: 1920×1080 với khung 640×360 -> 640×360; 1080×1920 (video dọc) -> 202×360.
    """
    h, w = frame_bgr.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return frame_bgr.copy()
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))  # cv2 nhận (W, H)
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def denoise(frame_bgr: np.ndarray, kernel: int) -> np.ndarray:
    """GaussianBlur để xóa nhiễu hạt/nhiễu nén trước khi phân ngưỡng màu.

    Không blur thì mask màu bị lốm đốm các chấm 1–2 pixel (nhiễu cảm biến, khối nén
    video, vân lá cây) -> TV3 phải lọc rất nhiều vùng rác.
    kernel <= 1 nghĩa là TẮT blur (dùng cho thí nghiệm có/không blur).
    Kernel chẵn sẽ được tăng lên số lẻ gần nhất vì OpenCV bắt buộc kích thước lẻ.
    """
    kernel = int(kernel)
    if kernel <= 1:
        return frame_bgr.copy()
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(frame_bgr, (kernel, kernel), 0)


def preprocess_color(frame_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Blur rồi đổi BGR -> HSV. Trả ảnh HSV uint8 cùng kích thước frame.

    Vì sao HSV: H (màu) tách khỏi V (độ sáng), nên ngọn lửa vẫn có H ~ đỏ-cam-vàng
    dù sáng hay tối; còn trong BGR cả 3 kênh cùng đổi khi độ sáng đổi, khó đặt ngưỡng.
    """
    check_frame(frame_bgr)
    return cv2.cvtColor(denoise(frame_bgr, cfg.blur_kernel), cv2.COLOR_BGR2HSV)


def to_ycrcb(frame_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Blur rồi đổi sang YCrCb (Y=độ sáng, Cr=độ đỏ, Cb=độ xanh lam) – dùng để so sánh."""
    check_frame(frame_bgr)
    return cv2.cvtColor(denoise(frame_bgr, cfg.blur_kernel), cv2.COLOR_BGR2YCrCb)


def to_gray(frame_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Blur rồi đổi sang ảnh xám (H×W) – dùng cho phân tích độ sáng/khói nếu cần."""
    check_frame(frame_bgr)
    return cv2.cvtColor(denoise(frame_bgr, cfg.blur_kernel), cv2.COLOR_BGR2GRAY)
