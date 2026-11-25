# Adaptive Region Motion cho mask khói

`adaptive_region_motion` đóng mask màu bằng ellipse 15×15, giãn mask chuyển động bằng ellipse 9×9 theo preset khói, rồi xét từng thành phần liên thông 8 hướng. Với thành phần chiếm không quá `region_max_area_ratio` của ảnh, thuật toán giữ nguyên quy tắc `region_motion`: diện tích tối thiểu 130 pixel, ít nhất 12 pixel chuyển động và tỷ lệ giao tối thiểu 0,015. Với thành phần lớn hơn ngưỡng, thuật toán chọn các vùng giao với chuyển động đủ lớn làm hạt giống, rồi mở rộng hạt giống trong mask màu tối đa 8 lần bằng kernel 5×5. Đặt `growth_iterations=0` để chạy lại cách chỉ giữ phần giao của Adaptive V1.

Mặc định `region_max_area_ratio=0.50` là giá trị thử nghiệm, chưa tối ưu. Mask chuyển động có nhiễu lớn vẫn có thể tạo vùng sai; vùng khói ít chuyển động hoặc hạt giống dưới 130 pixel vẫn bị bỏ. TV3 và bộ lọc thời gian có thể tiếp tục loại vùng dù Fusion đã giữ lại. Trong engine `opencv_context`, cảnh thường dùng Adaptive V2, cảnh trời dùng `sky_motion`, còn cảnh ám xanh và video dọc không có trời dùng `and_dilate`.

## Chạy A/B trên cùng video

```powershell
python -m fireguard.pipeline --input data/development/demo35.mp4 --output outputs/old.mp4 --csv outputs/old.csv --smoke-engine opencv_context --smoke-fusion region_motion --smoke-motion mog2 --smoke-motion-dilate 9 --smoke-vote-hits 5 --smoke-verify off --fire-fusion and_dilate --single-box-per-kind
python -m fireguard.pipeline --input data/development/demo35.mp4 --output outputs/new.mp4 --csv outputs/new.csv --smoke-engine opencv_context --smoke-fusion adaptive_region_motion --smoke-region-max-area-ratio 0.50 --smoke-motion mog2 --smoke-motion-dilate 9 --smoke-vote-hits 5 --smoke-verify off --fire-fusion and_dilate --single-box-per-kind
```

Không dùng số frame xác nhận SMOKE làm precision/recall khi chưa có ground truth.
