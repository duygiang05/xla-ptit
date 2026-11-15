# TV4 – Hướng dẫn hiểu và giải thích code

## TV4 làm gì?
TV3 trả về danh sách `Region(kind, bbox, area, centroid, score)` cho TỪNG frame. TV4 không đọc màu pixel trực tiếp: TV4 so sánh các vùng qua thời gian, và chỉ giữ vùng xuất hiện đủ thường xuyên.

- `features.py`: ghép ứng viên cùng loại dựa trên IoU hoặc khoảng cách tâm; tính `area_change`, `centroid_displacement`, `aspect_ratio_change`.
- `temporal_filter.py`: lưu lịch sử riêng cho từng vùng và từng nhãn, ghép 1-1 giữa vùng frame mới và vùng cũ, đếm số lần xuất hiện trong N frame gần nhất.
- `decision.py`: luật **K trong N frame**. Ví dụ `vote_window=5`, `vote_hits=3` nghĩa là có ít nhất 3 lần thấy vùng trong 5 frame thì trả kết quả. Các ngưỡng phải kiểm thử trước khi chốt.

## Ví dụ dễ giải thích
Frame 0: FIRE xuất hiện (1 hit) → chưa giữ.
Frame 1: FIRE xuất hiện (2 hits) → chưa giữ.
Frame 2: FIRE xuất hiện (3 hits) → giữ FIRE nếu K=3.
Nếu vùng chỉ thoáng xuất hiện 1 frame, bộ lọc loại được vùng đó.

Một video có thể có nhiều vùng FIRE và SMOKE; TV4 **không gộp** thành một nhãn chung.

## Giới hạn phải nói rõ khi báo cáo
- Đây là bộ lọc theo thời gian đơn giản, **không có AI, không xác nhận đám cháy thật**.
- Hiện tại `Region` của TV3 **chưa có chu vi contour**. TV4 dùng diện tích, vị trí tâm và tỉ lệ khung bao; không được gọi tỉ lệ khung bao là chu vi. Khi TV3 bổ sung perimeter, có thể mở rộng features.
- Giá trị `min_iou`, `max_center_distance`, `vote_window`, `vote_hits` trong `config.py` **chưa được tối ưu thực nghiệm**.
- Những đặc trưng tính được hiện được dùng để mô tả sự thay đổi và kiểm tra ứng viên tương ứng; **luật xác nhận hiện chỉ dùng Temporal Voting**, chưa có ngưỡng biến thiên hình dạng. Muốn dùng ngưỡng biến thiên, phải thực nghiệm trên video trước.
- Có thể nhầm vùng nếu hai đối tượng cùng loại ở quá gần; camera rung và khói quá mờ có thể gây lỗi. Không thay thế hệ thống báo cháy chuyên dụng.

## Chạy test
Mở Terminal tại thư mục `FireGuard`:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Thử thay `vote_window=5`, `vote_hits=3` trong `Config` ở một bài test để quan sát khác biệt.

## Nguyên tắc làm nhóm
Bạn chỉ cần thay ba file trong `fireguard/temporal/`, `tests/test_temporal.py`; module TV1, TV2, TV3 vẫn thuộc các thành viên khác. `pipeline.py` cũ đã gọi đúng `TemporalFilter.update(regions, frame_idx)` nên **không phải đổi API**.
