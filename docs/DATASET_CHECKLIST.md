# Dataset – Những việc cần chốt bằng dữ liệu thực

- [ ] Video thuộc bốn tình huống: FIRE-only, SMOKE-only, BOTH, NEITHER (âm tính gồm ánh đèn, hoàng hôn, mây, hơi nước).
- [ ] Mỗi video một hàng trong `data/manifest.csv`; `has_fire`/`has_smoke` là hai nhãn độc lập (0/1); không sao chép cùng video vào hai lớp.
- [ ] Ghi nguồn, quyền sử dụng, loại camera, độ phân giải, fps, chất lượng nén, ánh sáng.
- [ ] Chia development/integration/test theo **nguồn video gốc** để tránh trùng dữ liệu.
- [ ] Nếu cần đánh giá thời điểm phát hiện, thêm nhãn `data/annotations/events_template.csv` cho sự kiện FIRE/SMOKE tương ứng.
- [ ] Quy định TP/FP/FN trước khi công bố precision/recall; test cuối chỉ dùng sau khi khóa cấu hình.
- [ ] Không push video dung lượng lớn lên GitHub.
