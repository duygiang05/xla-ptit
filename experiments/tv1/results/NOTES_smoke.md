# TV1 – Thí nghiệm 2: luật màu khói (cảnh tổng hợp)

Lệnh: `python experiments/tv1/compare_smoke.py` · Dữ liệu: `smoke_scene.png` (640×360)
Khói vẽ bán trong suốt: `ảnh = (1-α)·nền + α·màu_khói`; GT khói = α ≥ 0.3, dày = α ≥ 0.6, mỏng = 0.3–0.6.
Tham số mặc định: `smoke_s_max=70`, `smoke_v` 65–230, `smoke_rgb_spread_max=30`, `blur_kernel=5`.
Bảng: `smoke_rules.csv`, `smoke_s_sweep.csv` · Ảnh: `smoke_rules_grid.png`.

## So sánh luật (tỉ lệ pixel bị đánh dấu)
| mode | lọc mịn | recall | dày | mỏng | precision | bê tông | sỏi | mây | trời | cây | lửa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hsv | tắt | 0.65 | 1.00 | 0.39 | 0.13 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| rgb | tắt | 0.33 | 0.73 | 0.02 | 0.07 | 0.99 | 0.99 | 0.97 | 0.00 | 0.00 | 0.00 |
| hsv | ≤ 12 | 0.65 | 1.00 | 0.38 | 0.16 | 1.00 | 0.54 | 0.86 | 0.00 | 0.00 | 0.00 |

(`and` trùng `rgb`, `or` trùng `hsv` trên cảnh này.)

## Quét `smoke_s_max` (luật HSV)
| S_max | 40 | 55 | 70 | 85 | 100 | 115 | 130 |
|---|---|---|---|---|---|---|---|
| recall khói mỏng | 0.01 | 0.05 | 0.39 | 0.83 | 0.97 | 0.97 | 0.97 |
| bắt nhầm trời | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.88 | 1.00 |

## Nhận xét
1. **Màu không phân biệt được khói với tường xám, sỏi, mây**: mọi luật đều đánh dấu gần
   100% các vùng này, nên precision chỉ 0.07–0.16. Mask khói bắt buộc phải kết hợp với chuyển
   động (TV2) và thời gian (TV4). Điều này khớp với docs: mask màu khói chỉ là gợi ý.
2. **Khói mỏng là điểm yếu chính**: khói bán trong suốt mang màu nền (trời xanh) nên S tăng,
   và luật HSV mặc định chỉ bắt 39% khói mỏng. Nới `smoke_s_max` lên 85–100 đưa mức này lên
   0.83–0.97. Ngưỡng 100 chỉ "an toàn" vì trời tổng hợp có S = 112; trời thật nhạt hơn
   (S ≈ 40–80) sẽ bị bắt nhầm. **Giữ mặc định 70; chốt lại trên video development.**
3. **Luật RGB-xám (chênh lệch tuyệt đối ≤ 30) chặt hơn HSV với pixel sáng**: ở V ≈ 190 thì
   S ≤ 70 tương đương chênh lệch ≤ 52. Vì vậy luật RGB chỉ giúp được khói tối (V < ~110) và
   bỏ sót khói nhạt. Giữ mặc định `hsv`; `or` không có hại nhưng ở đây cũng không có lợi.
4. **Lọc độ mịn (std ≤ 12)** loại được 46% sỏi có vân và 14% mây, chỉ mất ~1% khói mỏng,
   nhưng không loại được tường phẳng. Có lợi nhỏ; để mặc định tắt cho tới khi thử trên video
   thật có khói phủ lên nền có vân.

## Hạn chế
Khói thật đa dạng hơn (ngả vàng/nâu, dày mỏng thay đổi theo gió). Không chốt ngưỡng trên cảnh
này; không dùng `data/test/`.
