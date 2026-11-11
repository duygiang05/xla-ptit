"""TV3 – Hình thái học trên mask nhị phân (0/255).

- Opening (erode -> dilate): xóa chấm nhiễu nhỏ hơn phần tử cấu trúc.
- Closing (dilate -> erode): lấp lỗ nhỏ, nối các mảnh gần nhau của cùng vùng.
`clean_mask` (opening rồi closing) là hàm mặc định pipeline đang dùng.
"""
import cv2
import numpy as np

from fireguard.config import Config

_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}
_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
}


def make_kernel(size: int, shape: str = "ellipse") -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(_SHAPES[shape], (size, size))


def apply_morphology(mask: np.ndarray, ops: tuple[str, ...] = ("open", "close"),
                     ksize: int = 3, shape: str = "ellipse", iterations: int = 1) -> np.ndarray:
    """Áp dụng lần lượt các phép hình thái; `ops=()` trả về bản sao mask gốc."""
    kernel = make_kernel(ksize, shape)
    out = mask.copy()
    for op in ops:
        out = cv2.morphologyEx(out, _OPS[op], kernel, iterations=iterations)
    return out


def clean_mask(mask, cfg: Config):
    return apply_morphology(mask, ("open", "close"), cfg.morph_kernel, "ellipse")


def combine_masks(color_mask: np.ndarray, motion_mask: np.ndarray, mode: str = "and",
                  motion_dilate: int = 0) -> np.ndarray:
    """Kết hợp mask màu với mask chuyển động.

    mode='and'  : giao cứng (baseline của pipeline).
    mode='or'   : hợp, giữ nhiều ứng viên hơn (nhiều báo sai hơn).
    mode='color': chỉ dùng màu (để làm ablation).
    motion_dilate > 0: nở mask chuyển động trước khi giao, vì frame difference chỉ
    bật ở biên vùng chuyển động -> giao cứng làm vùng lửa/khói bị thủng.
    """
    if mode == "color":
        return color_mask.copy()
    if motion_dilate > 0:
        motion_mask = cv2.dilate(motion_mask, make_kernel(motion_dilate))
    if mode == "and":
        return cv2.bitwise_and(color_mask, motion_mask)
    if mode == "or":
        return cv2.bitwise_or(color_mask, motion_mask)
    raise ValueError(f"Unknown combine mode: {mode}")
