# Task: Cải tiến Smoke Fusion – FireGuard

Repo: https://github.com/duygiang05/xla-ptit

## Mục tiêu

Chỉ sửa Fusion của nhánh khói. Demo35 gặp tình trạng mask màu gần kín ảnh, khiến `region_motion` giữ cả nền khi một phần nhỏ có chuyển động. Bổ sung chế độ `adaptive_region_motion`, giữ nguyên chế độ cũ để so sánh A/B.

## 1. Thuật toán

Đọc `fireguard/fusion/mask_fusion.py` trước khi sửa. Tái sử dụng các bước hiện tại: closing mask màu (ellipse 15×15), dilation mask động (ellipse 9×9 theo preset khói), connected components 8 hướng.

Với mỗi thành phần màu C, tính Q = area(C)/(H×W), N = số pixel giao với mask động đã giãn, R = N/area(C).

- Nếu Q <= `region_max_area_ratio` (mặc định 0.50): giữ nguyên logic Region Motion (area >= 130, N >= 12, R >= 0.015).
- Nếu Q > ngưỡng: chỉ giữ phần giao C AND mask động đã giãn. Không bắt buộc R >= 0.015 trên toàn vùng lớn. Yêu cầu N >= 12; tách các thành phần liên thông của phần giao và loại vùng nhỏ hơn `region_min_area` (130). Tái sử dụng các tham số hiện có, không hardcode lại ngưỡng.
- Đầu ra giữ nguyên hợp đồng: ndarray 2D, uint8, 0/255, cùng kích thước đầu vào; không sửa mask đầu vào.
- Kiểm tra `region_max_area_ratio` thuộc (0, 1]. Ngưỡng 0.50 chỉ là giá trị thử nghiệm, chưa được tối ưu.

## 2. Phạm vi chỉnh sửa

- `fireguard/fusion/mask_fusion.py`: thêm mode và tham số mới, giữ nguyên hành vi các mode cũ.
- `fireguard/pipeline.py`: tích hợp mode và tham số CLI `--smoke-region-max-area-ratio`. Trong engine `opencv_context`, chỉ thay Fusion của cảnh thông thường; bảo đảm có thể chọn lại `region_motion` để so sánh trên cùng engine. Không chỉ sửa preset vì pipeline hiện chọn mode theo bối cảnh.
- `fireguard/presets.py`: đặt mode mới cho cảnh thông thường trên app; giữ nguyên các cấu hình khác. Chỉ sửa `app.py` hoặc export của package Fusion nếu thật sự cần.
- Giữ nguyên Fusion cảnh trời (`color_only`), ám xanh (`and_dilate`, 21), video dọc không có trời (`and_dilate`, 13).
- Tuyệt đối không sửa thuật toán TV1, TV2, TV3, tracking, temporal voting hay nhánh lửa. Không thêm dependency, không refactor ngoài phạm vi, không hardcode theo tên video/frame.

## 3. Kiểm thử

Bổ sung test trong `tests/test_fusion.py` và test tích hợp nếu cần:

- Vùng bình thường cho kết quả giống mode cũ; vùng lớn chỉ giữ phần giao.
- Vùng lớn có R < 1.5% vẫn có thể giữ phần giao đủ diện tích; không có chuyển động hoặc phần giao quá nhỏ thì loại.
- Nhiều thành phần, giá trị Q đúng ngưỡng, tham số không hợp lệ, đầu vào không bị sửa, đầu ra đúng kiểu.
- Các mode cũ và nhánh lửa không thay đổi; chạy `python -m pytest -q`.

Nếu có demo35, chạy A/B cùng engine, cùng video và cùng mọi tham số ngoại trừ mode Fusion. Xuất mask màu, mask động, Fusion cũ/mới và kết quả sau TV3 ở vài frame; thống kê tỷ lệ diện tích Fusion, số frame fallback, số frame có vùng sau TV3 và số frame xác nhận SMOKE. Kiểm thử hồi quy trên video dương/âm tính có sẵn. Không dùng tập test để dò ngưỡng, không báo cáo precision/recall nếu thiếu ground truth. Nếu thiếu video, báo rõ và dùng mask tổng hợp để test.

## 4. Kết quả cần bàn giao

Triển khai trực tiếp, kiểm tra git diff, báo danh sách file sửa, kết quả test, kết quả A/B thực tế (nếu có), và hai câu lệnh CLI chạy mode cũ/mới. Ghi ngắn nguyên lý và hạn chế vào `docs/ADAPTIVE_REGION_MOTION.md`.

Nếu Fusion đã khoanh vùng hợp lý nhưng TV3 vẫn loại, báo nguyên nhân cho thành viên TV3; không tự ý sửa phần của họ. Mục tiêu là cải thiện mask Fusion, không ép demo35 phải phát hiện SMOKE bằng mọi giá.

Không tự commit hoặc push.
