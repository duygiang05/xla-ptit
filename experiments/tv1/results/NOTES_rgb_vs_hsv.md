# TV1 – Thí nghiệm 1: luật màu lửa RGB / HSV / AND / OR (cảnh tổng hợp)

Lệnh: `python experiments/tv1/compare_rgb_hsv.py` · Dữ liệu: `synthetic_scene.png` (640×360, nhiễu σ=12)
Tham số: `fire_h` 0–35 ∪ 170–179, `fire_s_min=100`, `fire_v_min=125`, `fire_rgb_r_min=130`.
Bảng đầy đủ: `rgb_vs_hsv.csv` · Ảnh: `rgb_vs_hsv_grid.png`.

| mode | blur | recall lửa | precision | đốm nhiễu | đèn đỏ | sodium | áo đỏ | hoàng hôn | bê tông |
|---|---|---|---|---|---|---|---|---|---|
| hsv | 5 | 0.94 | 0.41 | 3 | 1.00 | 0.99 | 0.73 | 0.98 | 0.01 |
| rgb | 5 | 0.67 | 0.44 | 388 | 0.45 | 1.00 | 0.00 | 0.72 | 0.01 |
| and | 5 | 0.63 | 0.44 | 9 | 0.45 | 0.99 | 0.00 | 0.72 | 0.00 |
| or  | 5 | 0.99 | 0.41 | 378 | 1.00 | 1.00 | 0.73 | 0.98 | 0.02 |
| hsv | tắt | 0.92 | 0.42 | 87 | 1.00 | 0.99 | 0.57 | 0.97 | 0.00 |
| rgb | tắt | 0.63 | 0.35 | 4971 | 0.47 | 1.00 | 0.00 | 0.72 | 0.09 |

## Nhận xét
1. **HSV + blur 5 là lựa chọn mặc định tốt nhất**: recall 0.94, gần như không có đốm nhiễu.
   RGB bỏ sót lớp đỏ thẫm ngoài cùng (pixel có G < B) nên recall chỉ 0.67, và trên tường xám
   (R≈G≈B) nhiễu cảm biến làm điều kiện R ≥ G > B đúng ngẫu nhiên, sinh hàng trăm đốm.
2. **Blur rất cần thiết**: số đốm nhiễu của HSV giảm từ 87 xuống 3, của RGB từ 4971 xuống 388.
   Recall vẫn giữ nguyên hoặc tăng nhẹ.
3. **Màu đơn thuần không loại được đèn đỏ, đèn sodium và hoàng hôn**: mọi luật đều đánh dấu
   ~100% các vùng này, nên precision chỉ khoảng 0.4. Phải dựa vào chuyển động (TV2: đèn và trời
   đứng yên) và thời gian (TV4). Luật RGB loại được áo đỏ ngả hồng (B > G) mà HSV không loại được.
4. Lõi lửa trắng-vàng (S thấp) tạo một **lỗ** giữa mask HSV. Có thể để closing của TV3 lấp, hoặc
   thử thêm luật "lõi sáng" ở thí nghiệm sau.

## Hạn chế
Cảnh tổng hợp có màu đơn giản hơn video thật. Ngưỡng phải được chốt trên `data/development/`,
không chốt trên cảnh này, và không dùng `data/test/`.
