# TV4 — Fusion và Temporal Decision

TV4 phụ trách **cả** `fireguard/fusion/` và `fireguard/temporal/`, đồng thời phối hợp phần tích hợp `fireguard/pipeline.py`.

## Fusion

`fireguard/fusion/mask_fusion.py` cung cấp `fuse_masks(color_mask,motion_mask,mode,...)`; hai mask phải cùng kích thước, `uint8`, nhị phân. Các mode hiện có: `and`, `and_dilate`, `color_only`, `region_motion`. Preset app dùng `and_dilate` cho FIRE và `region_motion` làm nền cho SMOKE; nhánh `opencv_context` có xử lý bối cảnh bổ sung. `and_dilate` dùng kernel giãn chuyển động, không phải bán kính pixel. `color_only` chỉ nên dùng đối chứng vì dễ tăng báo giả.

## Temporal

`fireguard/temporal/temporal_filter.py` dùng `RegionTracker` của TV3 và lịch sử quan sát để bỏ phiếu K/N. `TemporalFilter(cfg,decision_mode='vote').update(regions,frame_idx)` trả vùng đã xác nhận ở frame hiện tại. `decision_mode='hybrid'` kiểm thêm biến đổi hình dạng theo `decision.py`; ngưỡng hybrid là thử nghiệm, không phải preset app. Pipeline có bộ lọc riêng cho FIRE và SMOKE; app dùng vote cho cả hai, SMOKE với 5 hit/cửa sổ 7 frame. `last_decisions` chỉ phục vụ debug, không tự ghi CSV.

## Kiểm thử và giới hạn

```powershell
python -m pytest tests/test_fusion.py tests/test_temporal.py tests/test_temporal_hybrid.py tests/test_pipeline.py -q
python -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp
```

Thử FIRE, SMOKE, video âm tính, camera rung và video dọc sau khi sửa Fusion/Temporal. Một box mỗi loại/frame là phép gộp hiển thị, không chứng minh phân đoạn đúng. Lần đo lịch sử khói với box ONNX cho F1 0,528 ở IoU ≥ 0,5/25 video chỉ là độ giống nhãn giả; model và đầu ra đối chiếu đã dọn. Muốn kết luận độ chính xác thật cần nhãn độc lập và tập test chưa dùng chỉnh tham số.
