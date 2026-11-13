"""TV1 – Mask ứng viên màu KHÓI (H×W, uint8, 0/255).

Khói xám nhạt, bán trong suốt -> màu trùng tường, bê tông, mây, sương, hơi nước.
Mask này CHỈ LÀ GỢI Ý, không phải điều kiện đủ: chuyển động (TV2: khói bay lên/lan ra)
và thời gian (TV4: tồn tại lâu, biến đổi chậm) mới là bằng chứng chính.
"""
import cv2
import numpy as np

from fireguard.config import Config

SMOKE_MODES = ("hsv", "rgb", "and", "or")


def smoke_mask_hsv(hsv: np.ndarray, cfg: Config) -> np.ndarray:
    """Luật HSV: màu nhạt (S thấp) và độ sáng trung bình.

    - S <= smoke_s_max          : loại vật có màu rực (lửa, áo, cây, trời xanh đậm).
    - smoke_v_min <= V <= v_max : loại vùng quá tối (bóng đêm) và quá chói (đèn, nắng gắt).
    Hạn chế: S tính theo TỈ LỆ (max-min)/max nên pixel tối hơi lệch màu có S lớn -> bỏ sót khói đen.
    """
    lo = np.array([0, 0, cfg.smoke_v_min], dtype=np.uint8)
    hi = np.array([179, cfg.smoke_s_max, cfg.smoke_v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def smoke_mask_rgb(frame_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Luật "độ xám" RGB: ba kênh gần bằng nhau (chênh lệch TUYỆT ĐỐI nhỏ), độ sáng vừa phải.

    Khác HSV: dùng chênh lệch tuyệt đối nên không "phạt" pixel tối -> giữ được khói xám đậm.
    Độ sáng lấy theo max(B,G,R) để cùng thang với V của HSV.
    """
    mx = frame_bgr.max(axis=2).astype(np.int16)
    mn = frame_bgr.min(axis=2).astype(np.int16)
    ok = ((mx - mn) <= cfg.smoke_rgb_spread_max) & (mx >= cfg.smoke_v_min) & (mx <= cfg.smoke_v_max)
    return ok.astype(np.uint8) * 255


def local_std(gray: np.ndarray, win: int) -> np.ndarray:
    """Độ lệch chuẩn cục bộ trong cửa sổ win×win:  std = sqrt(E[x²] - E[x]²).

    Dùng boxFilter (trung bình cửa sổ) nên nhanh, không cần vòng lặp Python.
    Giá trị nhỏ = vùng mịn (khói, tường phẳng); lớn = vùng có vân (sỏi, lá, gạch, biên vật).
    """
    win = max(3, int(win) | 1)  # số lẻ >= 3
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win))
    mean_sq = cv2.boxFilter(g * g, -1, (win, win))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0))


def smoke_texture_mask(frame_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """255 ở vùng đủ mịn (std cục bộ <= smoke_texture_max). Loại bề mặt xám có vân."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return (local_std(gray, cfg.smoke_texture_win) <= cfg.smoke_texture_max).astype(np.uint8) * 255


def smoke_candidates(hsv: np.ndarray, cfg: Config, frame_bgr: np.ndarray | None = None) -> np.ndarray:
    """Kết hợp luật màu theo cfg.smoke_color_mode, rồi (nếu bật) lọc độ mịn.

    frame_bgr (ảnh BGR đã blur) cần cho luật RGB và lọc độ mịn; nếu không truyền
    sẽ đổi ngược từ HSV (sai số làm tròn nhỏ).
    """
    mode = cfg.smoke_color_mode
    if mode not in SMOKE_MODES:
        raise ValueError(f"smoke_color_mode phải thuộc {SMOKE_MODES}, nhận {mode!r}")
    need_bgr = mode != "hsv" or cfg.smoke_texture_max > 0
    if need_bgr and frame_bgr is None:
        frame_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if mode == "hsv":
        mask = smoke_mask_hsv(hsv, cfg)
    elif mode == "rgb":
        mask = smoke_mask_rgb(frame_bgr, cfg)
    else:
        combine = cv2.bitwise_and if mode == "and" else cv2.bitwise_or
        mask = combine(smoke_mask_hsv(hsv, cfg), smoke_mask_rgb(frame_bgr, cfg))

    if cfg.smoke_texture_max > 0:
        mask = cv2.bitwise_and(mask, smoke_texture_mask(frame_bgr, cfg))
    return mask
