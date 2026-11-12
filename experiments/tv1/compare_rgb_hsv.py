"""TV1 – Thí nghiệm 1: so sánh luật màu lửa RGB / HSV / AND / OR, có và không blur.

Chạy từ thư mục gốc repo:
    python experiments/tv1/compare_rgb_hsv.py

Dữ liệu: cảnh TỔNG HỢP 640x360 có ground truth từng vật (chưa có video thật).
  - 'flame'       : ngọn lửa nhiều lớp màu (lõi trắng-vàng -> vàng -> cam -> đỏ -> đỉnh đỏ thẫm)
  - 'traffic_red' : đèn giao thông đỏ          (nhiễu màu lửa)
  - 'sodium'      : quầng đèn đường sodium cam  (nhiễu màu lửa)
  - 'red_shirt'   : áo đỏ ngả hồng (B > G)      (nhiễu màu lửa)
  - 'sunset'      : dải trời hoàng hôn cam-hồng (nhiễu màu lửa)
  - 'concrete'    : tường bê tông xám
  - 'sky'         : trời xanh (nền)
Kết quả ghi vào experiments/tv1/results/:
  rgb_vs_hsv.csv      : recall/precision lửa + % pixel bị đánh dấu trên từng vật nhiễu
  rgb_vs_hsv_grid.png : ảnh gốc, ground truth và mask của từng phương pháp
Giới hạn: màu tổng hợp đơn giản hơn video thật -> chỉ dùng để hiểu hành vi luật,
KHÔNG dùng để chốt ngưỡng (phải chốt trên data/development).
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fireguard.color import color_candidates  # noqa: E402
from fireguard.config import Config  # noqa: E402

OUT = ROOT / "experiments" / "tv1" / "results"
H, W = 360, 640
MODES = ("hsv", "rgb", "and", "or")
BLURS = (1, 5)  # 1 = tắt blur
NEGATIVES = ("traffic_red", "sodium", "red_shirt", "sunset", "concrete", "sky")


def flame_layers(cx, cy, scale):
    """Các đa giác lồng nhau của ngọn lửa: (màu BGR, đa giác), vẽ từ ngoài vào trong."""
    def poly(w, h, lift=0):
        pts = [(-w, 0), (-0.7 * w, -0.6 * h), (-0.35 * w, -0.45 * h), (-0.2 * w, -1.0 * h),
               (0.05 * w, -0.55 * h), (0.3 * w, -0.85 * h), (0.55 * w, -0.4 * h), (w, 0),
               (0.6 * w, 0.35 * h), (-0.6 * w, 0.35 * h)]
        return np.array([(cx + x * scale, cy + (y - lift) * scale) for x, y in pts], np.int32)
    return [((60, 10, 200), poly(80, 150)),        # viền/đỉnh đỏ thẫm, hue ~172
            ((20, 30, 215), poly(70, 125)),        # đỏ
            ((0, 120, 255), poly(55, 95)),         # cam
            ((0, 210, 255), poly(38, 65)),         # vàng
            ((170, 245, 255), poly(20, 35, -5))]   # lõi trắng-vàng, S thấp


def make_scene(seed=0):
    """Trả (ảnh BGR có nhiễu, dict tên vật -> mask bool)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    labels = np.zeros((H, W), np.uint8)
    names = ["sky", "concrete", "sunset", "flame", "traffic_red", "sodium", "red_shirt"]
    lid = {n: i + 1 for i, n in enumerate(names)}

    img[:] = (200, 160, 110); labels[:] = lid["sky"]                      # trời xanh
    img[200:] = (130, 128, 125); labels[200:] = lid["concrete"]           # tường xám
    for y in range(0, 50):                                                # hoàng hôn cam -> hồng
        t = y / 49
        img[y, 330:] = (int(70 + 60 * t), int(150 - 50 * t), 245)
    labels[0:50, 330:] = lid["sunset"]

    flame = np.zeros((H, W), np.uint8)
    for col, pts in flame_layers(150, 300, 1.0):
        cv2.fillPoly(img, [pts], col)
        cv2.fillPoly(flame, [pts], 255)
    labels[flame > 0] = lid["flame"]

    cv2.rectangle(img, (395, 80), (445, 190), (40, 40, 40), -1)           # thân đèn giao thông
    cv2.circle(img, (420, 105), 18, (30, 30, 240), -1)                    # bóng đỏ
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (420, 105), 18, 255, -1)
    labels[m > 0] = lid["traffic_red"]

    m = np.zeros((H, W), np.uint8); cv2.circle(m, (560, 130), 35, 255, -1)  # quầng sodium
    img[m > 0] = (40, 150, 255); labels[m > 0] = lid["sodium"]

    cv2.rectangle(img, (470, 230), (540, 330), (90, 40, 190), -1)         # áo đỏ hồng
    labels[230:331, 470:541] = lid["red_shirt"]

    # Làm mềm biên như ảnh camera thật, rồi thêm nhiễu cảm biến
    img = cv2.GaussianBlur(img, (7, 7), 0)
    noise = rng.normal(0, 12, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, {n: labels == i for n, i in lid.items()}


def evaluate(mask, gt):
    on = mask > 0
    flame = gt["flame"]
    tp = np.count_nonzero(on & flame)
    row = {"flame_recall": tp / max(1, np.count_nonzero(flame)),
           "precision": tp / max(1, np.count_nonzero(on)),
           "speckles": count_speckles(mask)}
    for n in NEGATIVES:
        row[f"fp_{n}"] = np.count_nonzero(on & gt[n]) / max(1, np.count_nonzero(gt[n]))
    return row


def count_speckles(mask, max_area=10):
    """Số đốm nhiễu tí hon (<= max_area pixel) – thứ blur cần loại bỏ."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(np.sum(stats[1:, cv2.CC_STAT_AREA] <= max_area))


def imwrite_unicode(path, img):
    """cv2.imwrite KHÔNG ghi được đường dẫn có dấu tiếng Việt trên Windows (trả False, không báo lỗi).
    Mã hóa PNG trong bộ nhớ rồi để NumPy ghi file."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise IOError(f"Không mã hóa được ảnh: {path}")
    buf.tofile(str(path))


def label(img, text):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, gt = make_scene()
    rows, tiles = [], {}
    for blur in BLURS:
        for mode in MODES:
            cfg = replace(Config(), blur_kernel=blur, fire_color_mode=mode)
            mask = color_candidates(frame, cfg)["fire"]
            rows.append({"mode": mode, "blur_kernel": blur, **evaluate(mask, gt)})
            tiles[(mode, blur)] = mask

    with (OUT / "rgb_vs_hsv.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    gt_vis = np.zeros((H, W, 3), np.uint8)
    gt_vis[gt["flame"]] = (0, 0, 255)
    for n in ("traffic_red", "sodium", "red_shirt", "sunset"):
        gt_vis[gt[n]] = (0, 200, 255)
    top = [label(frame, "input (synthetic + noise)"), label(gt_vis, "GT: red=fire, yellow=distractor")]
    mid = [label(tiles[(m, 5)], f"{m} | blur 5") for m in MODES]
    bot = [label(tiles[(m, 1)], f"{m} | no blur") for m in MODES]
    blank = np.zeros_like(top[0])
    grid = np.vstack([np.hstack(top + [blank, blank]), np.hstack(mid), np.hstack(bot)])
    imwrite_unicode(OUT / "rgb_vs_hsv_grid.png",
                    cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    imwrite_unicode(OUT / "synthetic_scene.png", frame)

    cols = ["mode", "blur_kernel", "flame_recall", "precision", "speckles"] + [f"fp_{n}" for n in NEGATIVES]
    print(" | ".join(c.replace("fp_", "") for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    print(f"\nĐã lưu kết quả vào {OUT}")


if __name__ == "__main__":
    main()
