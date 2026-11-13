"""TV1 – Thí nghiệm 3 (phần bổ sung): so sánh 4 luật màu hsv / rgb / and / or trên VIDEO THẬT (data/development).

Chạy từ thư mục gốc repo:
    python experiments/tv1/compare_modes_real.py

Bổ sung cho Thí nghiệm 1-2 (chỉ chạy trên ảnh tổng hợp) và Thí nghiệm 3 (chỉ quét hsv và and cho lửa).
Giữ nguyên mọi ngưỡng khác ở Config() hiện tại, chỉ đổi fire_color_mode / smoke_color_mode,
nên sự khác biệt giữa các dòng là do luật màu gây ra.

Dữ liệu và cách đo giống sweep_dev.py (280 frame lấy mẫu từ 14 video development, nhãn mức video):
  recall, fpr, cover_pos, cover_neg, speckles_per_frame.
Lửa: chạy ở blur 5 (mặc định). Khói: chạy có và không lọc độ mịn (smoke_texture_max).
Không dùng data/test.
Kết quả: experiments/tv1/results/sweep/modes_real.csv
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_dev import OUT, evaluate, load_dev_frames, write  # noqa: E402
from fireguard.config import Config  # noqa: E402

MODES = ("hsv", "rgb", "and", "or")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    data = load_dev_frames(cfg)
    print(f"{len(data)} video, {sum(len(v['frames']) for v in data)} frame\n")

    rows = []
    for mode in MODES:
        rows.append({"kind": "fire", "mode": mode, "texture": "-",
                     **evaluate(data, replace(cfg, fire_color_mode=mode), "fire")})
    for tex in (0.0, cfg.smoke_texture_max):
        for mode in MODES:
            rows.append({"kind": "smoke", "mode": mode, "texture": tex,
                         **evaluate(data, replace(cfg, smoke_color_mode=mode, smoke_texture_max=tex), "smoke")})
    write("modes_real.csv", rows)

    print(f"{'kind':5s} {'mode':4s} {'tex':>5s} | recall |  fpr | phủ có | phủ không | đốm/frame")
    for r in rows:
        print(f"{r['kind']:5s} {r['mode']:4s} {str(r['texture']):>5s} | {r['recall']:6.2f} | {r['fpr']:.2f} | "
              f"{r['cover_pos']:6.1%} | {r['cover_neg']:9.1%} | {r['speckles_per_frame']:.1f}")
    print(f"\nĐã lưu vào {OUT / 'modes_real.csv'}")


if __name__ == "__main__":
    main()
