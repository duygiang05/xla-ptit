"""TV1 – Mask ứng viên màu LỬA (H×W, uint8, 0/255).

Mục tiêu là RECALL CAO (ít bỏ sót); các vùng có màu giống lửa nhưng không phải lửa
(đèn đỏ/cam, đèn sodium, hoàng hôn, xe/áo đỏ) sẽ được TV2 (chuyển động) và
TV4 (thời gian) lọc tiếp. Mask này chỉ là ỨNG VIÊN, không phải kết luận "có cháy".
"""
import cv2
import numpy as np

from fireguard.config import Config
from .preprocessing import check_frame, denoise
from .smoke_detection import smoke_candidates

FIRE_MODES = ("hsv", "rgb", "and", "or")


def fire_mask_hsv(hsv: np.ndarray, cfg: Config) -> np.ndarray:
    """Luật HSV: hue đỏ-cam-vàng (2 khoảng vì đỏ nằm ở 2 đầu thang 0–179), đủ đậm, đủ sáng.

    - S >= fire_s_min : loại tường xám/bê tông, mây, ánh sáng trắng (màu nhạt).
    - V >= fire_v_min : loại vùng tối (bóng râm, vật đỏ trong tối); lửa tự phát sáng nên V cao.
    - hue             : loại trời xanh, lá cây, nước.
    KHÔNG loại được: đèn đỏ/cam, đèn sodium, nắng chiều/hoàng hôn, xe/áo đỏ sáng.
    """
    s_v_lo = (cfg.fire_s_min, cfg.fire_v_min)
    lo1 = np.array([cfg.fire_h_low, *s_v_lo], dtype=np.uint8)
    hi1 = np.array([cfg.fire_h_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1)
    if cfg.fire_h_wrap_low <= 179:  # khoảng đỏ thẫm 170–179
        lo2 = np.array([cfg.fire_h_wrap_low, *s_v_lo], dtype=np.uint8)
        hi2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    return mask


def fire_mask_rgb(frame_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Luật RGB cổ điển (Chen et al., 2004): R >= R_T  và  R >= G > B.

    Ý tưởng: trong lửa, đỏ luôn mạnh nhất, xanh lam yếu nhất (đỏ -> cam -> vàng).
    - R >= R_T      : loại vùng tối.
    - R >= G > B    : loại vải/sơn đỏ ngả hồng/tím (B > G), trời, lá cây, tường xám (R≈G≈B).
    KHÔNG loại được: đèn đỏ/cam, đèn sodium, hoàng hôn (cũng có R >= G > B).
    """
    # Đổi sang int16 để phép so sánh không bị tràn số uint8.
    b, g, r = (frame_bgr[:, :, i].astype(np.int16) for i in range(3))
    ok = (r >= cfg.fire_rgb_r_min) & (r >= g) & (g > b)
    return ok.astype(np.uint8) * 255


def fire_candidates(hsv: np.ndarray, cfg: Config, frame_bgr: np.ndarray | None = None) -> np.ndarray:
    """Kết hợp luật màu theo cfg.fire_color_mode.

    frame_bgr (ảnh BGR đã blur) chỉ cần khi mode dùng luật RGB; nếu không truyền,
    sẽ đổi ngược từ HSV (sai số làm tròn nhỏ, chấp nhận được).
    """
    mode = cfg.fire_color_mode
    if mode not in FIRE_MODES:
        raise ValueError(f"fire_color_mode phải thuộc {FIRE_MODES}, nhận {mode!r}")
    if mode == "hsv":
        return fire_mask_hsv(hsv, cfg)
    if frame_bgr is None:
        frame_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = fire_mask_rgb(frame_bgr, cfg)
    if mode == "rgb":
        return rgb
    combine = cv2.bitwise_and if mode == "and" else cv2.bitwise_or
    return combine(fire_mask_hsv(hsv, cfg), rgb)


def color_candidates(frame_bgr: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """API của TV1 (docs/INTERFACES.md): trả {'fire': mask, 'smoke': mask}, cùng kích thước frame.

    Hai mask độc lập: một pixel/vùng có thể là ứng viên của cả hai.
    Blur 1 lần, dùng chung cho cả nhánh lửa và khói.
    """
    check_frame(frame_bgr)
    blurred = denoise(frame_bgr, cfg.blur_kernel)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return {'fire': fire_candidates(hsv, cfg, blurred), 'smoke': smoke_candidates(hsv, cfg, blurred)}
