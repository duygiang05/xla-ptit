from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Placeholder defaults: tune on development videos, not on held-out test.
    width: int = 640
    height: int = 360
    blur_kernel: int = 5
    fire_h_low: int = 0
    fire_h_high: int = 35  # TV1: 25 -> 35 để không bỏ sót lõi lửa vàng (vàng thuần = 30)
    fire_s_min: int = 100
    fire_v_min: int = 125
    # TV1 – hue đỏ "quấn vòng" ở cuối thang 0–179: lấy thêm [fire_h_wrap_low, 179].
    #   Giảm (vd 160) để bắt lửa đỏ thẫm/hồng hơn; đặt 180 để tắt khoảng này.
    fire_h_wrap_low: int = 170
    # TV1 – luật RGB cổ điển: R >= fire_rgb_r_min và R >= G > B.
    #   Tăng nếu bắt nhầm nhiều vùng đỏ tối; giảm nếu bỏ sót lửa trong cảnh tối.
    fire_rgb_r_min: int = 130
    # TV1 – cách kết hợp luật màu lửa: 'hsv' | 'rgb' | 'and' (chặt hơn) | 'or' (recall cao hơn).
    fire_color_mode: str = "hsv"
    smoke_s_max: int = 70
    smoke_v_min: int = 65
    smoke_v_max: int = 230
    # TV1 – luật "độ xám" RGB: max(B,G,R) - min(B,G,R) <= ngưỡng (khói gần như không màu).
    #   Tăng nếu khói ngả vàng/xanh bị bỏ sót; giảm nếu bắt nhầm nhiều vật nhạt màu.
    smoke_rgb_spread_max: int = 30
    # TV1 – lọc độ mịn: độ lệch chuẩn cục bộ của ảnh xám <= ngưỡng (khói làm nhòe nền).
    #   0 = TẮT. Bật (vd 12) để loại sỏi/đá/nhựa đường xám có vân; quá nhỏ sẽ bỏ sót khói mỏng.
    smoke_texture_max: float = 0.0
    smoke_texture_win: int = 9  # cửa sổ tính độ lệch chuẩn (pixel, số lẻ)
    # TV1 – kết hợp luật màu khói: 'hsv' | 'rgb' | 'and' | 'or'.
    smoke_color_mode: str = "hsv"
    motion_threshold: int = 18
    min_region_area: int = 130
    morph_kernel: int = 3
    vote_window: int = 7
    vote_hits: int = 3
    min_iou: float = 0.1
    max_center_distance: float = 55.0
