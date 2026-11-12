# TV1 – Thí nghiệm 4: chỉ màu vs màu + chuyển động (chuỗi tổng hợp)

Lệnh: `python experiments/tv1/color_vs_motion.py` · 60 frame 640×360, bỏ 10 frame đầu (khởi động).
Dùng NGUYÊN code TV2 (`MotionDetector` = frame difference ngưỡng 18; `MOG2MotionDetector`) và TV3
(`combine_masks`, `extract_regions`) với tham số mặc định – không sửa, không chỉnh.
Bảng: `color_vs_motion.csv` · Ảnh: `color_vs_motion_grid.png` (frame 35).

Chỉ số: **pixel recall** = tỉ lệ pixel lửa/khói giữ lại (trung bình theo frame);
**frames found** = tỉ lệ frame có ≥ 1 vùng (`extract_regions`) mà ≥ 30% bbox là lửa/khói thật;
**vùng nhầm/frame** = số vùng không thỏa điều kiện trên.

## Nhánh LỬA
| cách | pixel recall | frames found | vùng nhầm/frame | đèn đỏ | sodium | xe đỏ chạy |
|---|---|---|---|---|---|---|
| chỉ màu | 0.94 | 1.00 | 2.24 | 1.00 | 0.99 | 0.92 |
| màu ∧ frame diff (pipeline) | 0.28 | 0.94 | 0.26 | 0.00 | 0.00 | 0.12 |
| màu ∧ frame diff nở 15 px | 0.79 | 1.00 | 0.62 | 0.00 | 0.00 | 0.41 |
| màu ∧ MOG2 | 0.11 | 0.46 | 0.34 | 0.00 | 0.00 | 0.29 |

## Nhánh KHÓI (frames found / pixel recall)
| cách | khói nhạt, trơn | khói nhạt, có vân | khói đen, có vân |
|---|---|---|---|
| chỉ màu | 0.00 / 0.82 | 0.78 / 0.81 | 0.88 / 0.70 |
| màu ∧ frame diff | 0.00 / 0.00 | 0.10 / 0.00 | 0.38 / 0.05 |
| màu ∧ frame diff nở 15 px | 0.18 / 0.00 | 0.32 / 0.01 | **1.00 / 0.35** |
| màu ∧ MOG2 | 0.00 / 0.00 | 0.00 / 0.01 | 0.02 / 0.02 |

Tường và mây đứng yên: từ ~100% xuống ≤ 5% khi ghép với chuyển động (mọi kịch bản).

## Nhận xét
1. **Chuyển động loại sạch báo nhầm đứng yên mà màu không loại được**: đèn đỏ, đèn sodium,
   tường, mây giảm từ ~100% xuống ~0%. Vùng nhầm của nhánh lửa giảm từ 2.24 xuống 0.26/frame.
   Đây là bằng chứng cho thiết kế "màu recall cao, chuyển động lọc tiếp".
2. **Xe đỏ đang chạy KHÔNG bị loại** (vẫn 12–41%): vừa đúng màu vừa có chuyển động. Cần
   hình dạng (TV3: xe là khối chữ nhật cứng, lửa răng cưa và đổi hình liên tục) hoặc thời gian (TV4).
3. **Giao cứng với frame difference làm thủng vùng lửa** (pixel recall 0.94 → 0.28) vì frame
   difference chỉ bật ở biên vùng đổi hình. Vùng lửa vẫn được tìm thấy ở 94% frame, nhưng bbox
   bị vụn. Nở mask chuyển động 15 px lấy lại recall 0.79, đổi lại xe đỏ bị giữ nhiều hơn.
4. **Khói là trường hợp khó nhất với phép giao cứng**: khói bay chậm, mép mờ, và khói nhạt có
   độ sáng gần bằng trời, nên frame difference trên ảnh xám gần như không thấy. Chỉ khói tương phản
   cao, có vân cuộn, kèm mask chuyển động đã nở, mới được tìm thấy ổn định. Đề xuất cho nhóm
   trưởng/TV2 (không tự sửa): cân nhắc với nhánh khói dùng `motion_dilate`, hoặc tích lũy chuyển
   động qua nhiều frame, hoặc ngưỡng chuyển động riêng thấp hơn – đúng như cảnh báo trong
   `docs/INTERFACES.md` ("nhất là với khói ít chuyển động").
5. **Báo nhầm mới của nhánh khói sau khi ghép**: viền tường xám lộ ra khi lửa co lại hoặc xe
   chạy qua (~1.2 vùng/frame) – vừa xám vừa "chuyển động".
6. MOG2 với tham số mặc định học nhanh ngọn lửa lập lòe thành nền (frames found 0.46) trong
   chuỗi ngắn này; đây là quan sát cho TV2, không phải kết luận về MOG2 nói chung.

## Hạn chế
- Khói tổng hợp: kết quả phụ thuộc mạnh vào độ vân và độ tương phản do chính thí nghiệm đặt ra.
  Kịch bản "trơn" là đối chứng cho thấy điều này. Vân xoáy chỉ được điều chỉnh MỘT lần (từ quá
  yếu, std 0.12, lên mức cuộn rõ) rồi giữ cố định để tránh "chỉnh dữ liệu cho ra kết quả mong muốn".
- Phải chạy lại trên video thật trong `data/development/` trước khi đưa kết luận về khói vào báo cáo.
