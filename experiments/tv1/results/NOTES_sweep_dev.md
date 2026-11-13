# TV1 – Thí nghiệm 3: quét ngưỡng màu trên video thật (data/development)

Lệnh: `python experiments/tv1/sweep_dev.py` (bảng trong `results/sweep/`) và
`python experiments/tv1/visualize_frames.py [--set k=v ...] --tag <tên>` (ảnh trong `results/real_frames/`),
`python experiments/tv1/compare_modes_real.py` (so sánh 4 luật màu, `sweep/modes_real.csv`).
Dữ liệu: 14 video development của FIRESENSE (CC BY 4.0), 20 frame rải đều mỗi video, tổng 280 frame,
resize giữ tỉ lệ ≤ 640×360. **Không dùng `data/test`.**

## Cách đo (chỉ có nhãn mức video)
- Một frame **báo** nếu mask có thành phần liên thông ≥ `min_region_area` (130 px).
- **recall**: tỉ lệ frame báo trong video có lửa/khói. **fpr**: tỉ lệ frame báo trong video không có.
- **phủ có / phủ không**: % pixel mask bật, trung bình ở video có / không có.
- Hạn chế: video "có khói" cũng có trời, tường → recall là cận trên lạc quan. Vì vậy mọi cấu hình
  ứng viên đều được **xem bằng mắt** (`real_frames/*.jpg`) trước khi chọn.

## Kết quả: mốc trước khi chỉnh và cấu hình cuối (`sweep/final_vs_baseline.csv`)
| | recall | fpr | phủ có | phủ không |
|---|---|---|---|---|
| lửa – mốc | 1.00 | 0.27 | 18.7% | 0.7% |
| **lửa – cuối** (+ lõi sáng) | 1.00 | 0.27 | **20.6%** | 0.7% |
| khói – mốc (`s_max=70`, không lọc mịn) | 1.00 | 0.80 | 71.5% | 43.5% |
| **khói – cuối** (`s_max=30`, lọc mịn ≤ 12) | 1.00 | 0.78 | 32.4% | **11.4%** |

## Thay đổi mặc định trong `fireguard/config.py`
| tham số | cũ → mới | lý do |
|---|---|---|
| `smoke_s_max` | 70 → **30** | Trời nhạt, tường, sỏi thật có S ≈ 30–70 nên bị bật hàng loạt (`real_frames/*_baseline.jpg`). |
| `smoke_texture_max` | 0 → **12** | Loại sỏi, gạch, mặt đường có vân. Mức 8 cho fpr thấp hơn nhưng bắt đầu mất khói mỏng. |
| `fire_core_v_min` | (mới) **240**, `s_max=100`, `dilate=7` | Lấp lõi vàng-trắng bị thủng (`fs_fire_pos04_final.jpg`). Độ phủ lửa tăng 18.7% → 20.6%, fpr không đổi. Mức lỏng hơn (V ≥ 200 hoặc nở 15 px) làm fpr tăng tới 0.46. |
| `smoke_v_min` | **giữ 65** | Cấu hình tự động "tốt nhất" (`s_max=20, v_min=130`) làm **mất khói đen** (`fs_fire_pos01_smoke_tuned.jpg`), loại khói nguy hiểm nhất. Bác bỏ. |
| luật màu lửa | **giữ** | "Tốt nhất" tự động (`s_min=130, v_min=180, h_high=25, and`) giảm fpr 0.27 → 0.04 nhưng làm **mất nửa ngọn lửa** (phủ 18.7% → 9.5%), trái mục tiêu recall cao. Phương án `fire_s_min=130` (fpr 0.20, mất 10% độ phủ) để ngỏ cho nhóm quyết khi chạy cả pipeline. |

## Blur (`sweep/blur_ablation.csv`)
Số đốm nhiễu ≤ 10 px mỗi frame: lửa 31 → 9 → 5, khói 128 → 25 → 10 (kernel 1 → 5 → 9). recall và fpr
gần như không đổi. Giữ kernel 5: giảm nhiễu mạnh mà ít làm mờ chi tiết ngọn lửa nhỏ.

## So sánh 4 luật màu hsv / rgb / and / or (`sweep/modes_real.csv`)
Thí nghiệm 1–2 chỉ so 4 luật trên ảnh tổng hợp; lưới quét ở trên chỉ có `hsv` và `and` cho lửa.
Mục này chạy đủ 4 luật trên cùng 280 frame, giữ mọi ngưỡng ở mặc định cuối, chỉ đổi `*_color_mode`.

**Lửa** (blur 5, có lõi sáng)
| luật | recall | fpr | phủ có | phủ không | đốm/frame |
|---|---|---|---|---|---|
| **hsv** (chọn) | 1.00 | 0.27 | 20.6% | 0.73% | 9.9 |
| rgb | 1.00 | 1.00 | 29.1% | 16.2% | 35.6 |
| and | 1.00 | 0.18 | 18.4% | 0.44% | 8.1 |
| or | 1.00 | 1.00 | 31.2% | 16.5% | 37.2 |

**Khói** (blur 5; `and` trùng `hsv`, `or` trùng `rgb` vì luật RGB-xám chặt hơn HSV ở hầu hết pixel)
| luật | lọc mịn | fpr | phủ có | phủ không | đốm/frame |
|---|---|---|---|---|---|
| **hsv / and** (chọn) | 12 | 0.78 | 32.4% | **11.5%** | 38.0 |
| rgb / or | 12 | 0.80 | 48.5% | 20.6% | 39.8 |
| hsv / and | tắt | 0.80 | 47.0% | 20.1% | 44.3 |
| rgb / or | tắt | 0.80 | 67.7% | 40.4% | 23.8 |

So với ảnh tổng hợp (thí nghiệm 1): HSV vẫn được chọn nhưng **vì lý do khác**. Trên ảnh tổng hợp RGB thua vì
bỏ sót lửa (recall 0.67); trên video thật mọi luật đều recall 1.00, RGB thua vì **báo nhầm 100% frame
không có lửa** (cờ, xe, tường nắng thỏa R ≥ G > B). AND từ "kém nhất" trên ảnh tổng hợp thành đối thủ sát nút.

## Nhận xét
1. **Mask lửa tách khá tốt video có/không có lửa**: video âm tính chỉ bật 0–3.5% pixel. Nguồn báo nhầm
   là chữ đỏ trên cờ vàng, xe tải cam, quạt cam (`fs_smoke_pos02`), đúng loại "vật màu lửa" mà chuyển
   động/hình dạng phải lọc tiếp. Thân xe đỏ sẫm (`fs_fire_neg11`) bị loại nhờ ngưỡng V.
2. **Mask khói trên video thật kém hơn cảnh tổng hợp**: mặc định cũ bật 43.5% khung hình ở video âm tính.
   Sau khi chỉnh còn 11.4%, nhưng fpr frame vẫn 0.78, tức là gần như frame nào cũng còn một mảng
   "xám, mịn" (đường sỏi sáng, tường be, trời). **Khẳng định lại: mask màu khói chỉ là gợi ý**;
   khói phải được xác nhận bằng chuyển động và thời gian.
3. **Tự động chọn theo một chỉ số dễ sai**: cả hai cấu hình "tốt nhất" theo fpr đều bỏ sót thứ quan trọng
   (khói đen, nửa ngọn lửa). Cần xem ảnh, và giữ ràng buộc về độ phủ.
4. **HSV là luật đúng cho cả lửa và khói trên video thật**. RGB và OR bị loại hẳn (fpr lửa 1.00). AND giảm
   fpr lửa 0.27 → 0.18 nhưng mất ~11% độ phủ lửa (20.6% → 18.4%); vì mask màu ưu tiên recall và chuyển
   động/thời gian sẽ lọc tiếp, giữ HSV. AND là phương án dự phòng nếu cả pipeline còn quá nhiều báo nhầm.

## Ghi chú tái lập
- Kết quả thí nghiệm 1, 2, 4 (cảnh tổng hợp) được tạo với **mặc định cũ** (`smoke_s_max=70`, không lọc
  mịn, không lõi sáng), như ghi trong NOTES của từng thí nghiệm. Chạy lại bây giờ sẽ ra số khác cho
  nhánh khói.
- `real_frames/*_baseline.jpg`: mặc định cũ; `*_final.jpg`: mặc định mới; `*_smoke_tuned.jpg`: ứng viên
  bị bác (`s_max=20, v_min=130, texture=12`); `*_smoke_s30_tex12.jpg`: ứng viên được chọn (trước khi bật lõi lửa).
