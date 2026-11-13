"""TV1 – Thí nghiệm 2: luật màu khói HSV / RGB-xám / AND / OR, có và không lọc độ mịn.

Chạy từ thư mục gốc repo:
    python experiments/tv1/compare_smoke.py

Dữ liệu: cảnh TỔNG HỢP 640x360 có ground truth (chưa có video thật).
  Khói vẽ BÁN TRONG SUỐT bằng alpha blending:  ảnh = (1-α)·nền + α·màu_khói
  - 'smoke_light' : cột khói xám nhạt bay lên trước trời xanh và tán cây
  - 'smoke_dark'  : cột khói xám đậm (khói đen do cháy vật liệu)
  GT khói = α >= 0.3; chia 'dày' (α >= 0.6) và 'mỏng' (0.3 <= α < 0.6).
  Vật gây nhầm: tường bê tông phẳng, sỏi xám có vân, mây trắng-xám, trời xanh, cây, lửa.
Kết quả ghi vào experiments/tv1/results/:
  smoke_rules.csv, smoke_rules_grid.png, smoke_scene.png
Giới hạn: khói thật có hình dạng/màu đa dạng hơn -> không dùng cảnh này để chốt ngưỡng.
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fireguard.color import color_candidates  # noqa: E402
from fireguard.config import Config  # noqa: E402
from _utils import imwrite_unicode, label  # noqa: E402

OUT = ROOT / "experiments" / "tv1" / "results"
H, W = 360, 640
MODES = ("hsv", "rgb", "and", "or")
TEXTURES = (0.0, 12.0)  # 0 = tắt lọc độ mịn
NEGATIVES = ("concrete", "gravel", "cloud", "sky", "tree", "fire")


def plume_alpha(x0, y0, height, width, peak, rng):
    """Độ đặc α của cột khói: nhiều đốm Gauss chồng nhau, nở rộng và loãng dần khi bay lên."""
    alpha = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(14):
        t = i / 13                                   # 0 = chân cột, 1 = đỉnh
        cx = x0 + 25 * np.sin(3 * t) + rng.normal(0, 4)
        cy = y0 - t * height
        sigma = width * (0.35 + 0.65 * t)
        alpha += (1.0 - 0.6 * t) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(alpha / alpha.max() * peak, 0, 1)


def make_scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    region = {}

    def put(name, mask, color=None, pixels=None):
        if color is not None:
            img[mask] = color
        if pixels is not None:
            img[mask] = pixels[mask]
        for other in region:                     # vật vẽ sau che vật vẽ trước -> vùng không chồng nhau
            region[other] = region[other] & ~mask
        region[name] = mask

    full = np.ones((H, W), bool)
    put("sky", full, (205, 165, 115))                                   # trời xanh
    ground = np.zeros((H, W), bool); ground[230:] = True
    conc = ground.copy(); conc[:, 320:] = False
    put("concrete", conc, (128, 128, 126))                              # bê tông phẳng
    # Sỏi: các khối 3x3 xám ngẫu nhiên -> vân thô, vẫn còn sau blur 5x5
    blocks = rng.normal(125, 28, (H // 3 + 1, W // 3 + 1)).clip(0, 255)
    gravel_px = np.repeat(np.repeat(blocks, 3, 0), 3, 1)[:H, :W, None].repeat(3, 2)
    grav = ground.copy(); grav[:, :320] = False
    put("gravel", grav, pixels=gravel_px)
    cloud = np.zeros((H, W), np.uint8)
    cv2.ellipse(cloud, (540, 60), (85, 32), 0, 0, 360, 255, -1)
    cv2.ellipse(cloud, (490, 75), (55, 25), 0, 0, 360, 255, -1)
    put("cloud", cloud > 0, (215, 212, 208))                            # mây trắng-xám
    tree = np.zeros((H, W), np.uint8)
    cv2.circle(tree, (170, 150), 60, 255, -1)
    put("tree", tree > 0, (40, 120, 50))                                # tán cây
    fire = np.zeros((H, W), np.uint8)
    cv2.ellipse(fire, (260, 222), (28, 14), 0, 0, 360, 255, -1)
    put("fire", fire > 0, (0, 140, 255))                                # đốm lửa ở chân khói

    # Khói: trộn alpha lên nền (nhìn xuyên được -> khói mỏng mang màu nền)
    a_light = plume_alpha(255, 215, 190, 34, 0.85, rng)
    a_dark = plume_alpha(400, 225, 150, 26, 0.9, rng)
    for a, col in ((a_light, (178, 176, 172)), (a_dark, (78, 82, 86))):
        img = (1 - a[..., None]) * img + a[..., None] * np.array(col, np.float32)
    alpha = np.maximum(a_light, a_dark)
    smoke = alpha >= 0.3
    for n in list(region):                       # vùng bị khói che không còn là "nhiễu" nữa
        region[n] = region[n] & (alpha < 0.1)
    region["smoke"], region["smoke_dense"], region["smoke_thin"] = smoke, alpha >= 0.6, smoke & (alpha < 0.6)

    img = cv2.GaussianBlur(img, (5, 5), 0) + rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), region


def evaluate(mask, gt):
    on = mask > 0
    tp = np.count_nonzero(on & gt["smoke"])
    row = {"recall": tp / np.count_nonzero(gt["smoke"]),
           "recall_dense": np.count_nonzero(on & gt["smoke_dense"]) / np.count_nonzero(gt["smoke_dense"]),
           "recall_thin": np.count_nonzero(on & gt["smoke_thin"]) / np.count_nonzero(gt["smoke_thin"]),
           "precision": tp / max(1, np.count_nonzero(on))}
    for n in NEGATIVES:
        row[f"fp_{n}"] = np.count_nonzero(on & gt[n]) / max(1, np.count_nonzero(gt[n]))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, gt = make_scene()
    rows, tiles = [], {}
    for tex in TEXTURES:
        for mode in MODES:
            cfg = replace(Config(), smoke_color_mode=mode, smoke_texture_max=tex)
            mask = color_candidates(frame, cfg)["smoke"]
            rows.append({"mode": mode, "texture_max": tex, **evaluate(mask, gt)})
            tiles[(mode, tex)] = mask

    with (OUT / "smoke_rules.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    gt_vis = np.zeros((H, W, 3), np.uint8)
    gt_vis[gt["smoke_thin"]] = (120, 120, 120)
    gt_vis[gt["smoke_dense"]] = (255, 255, 255)
    for n in ("concrete", "gravel", "cloud"):
        gt_vis[gt[n]] = (0, 200, 255)
    blank = np.zeros_like(frame)
    top = [label(frame, "input (synthetic smoke)"), label(gt_vis, "GT: white=dense grey=thin yellow=distractor"),
           blank, blank]
    mid = [label(tiles[(m, 0.0)], f"{m} | no texture filter") for m in MODES]
    bot = [label(tiles[(m, 12.0)], f"{m} | texture<=12") for m in MODES]
    grid = np.vstack([np.hstack(top), np.hstack(mid), np.hstack(bot)])
    imwrite_unicode(OUT / "smoke_rules_grid.png",
                    cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    imwrite_unicode(OUT / "smoke_scene.png", frame)

    # Quét ngưỡng S của luật HSV: nới S bắt thêm khói mỏng (mang màu nền) nhưng trả giá thế nào?
    sweep = []
    for s_max in (40, 55, 70, 85, 100, 115, 130):
        cfg = replace(Config(), smoke_s_max=s_max)
        sweep.append({"smoke_s_max": s_max, **evaluate(color_candidates(frame, cfg)["smoke"], gt)})
    with (OUT / "smoke_s_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0]))
        writer.writeheader()
        for r in sweep:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    for table in (rows, sweep):
        cols = list(table[0])
        print(" | ".join(c.replace("fp_", "") for c in cols))
        for r in table:
            print(" | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
        print()
    print(f"Đã lưu kết quả vào {OUT}")


if __name__ == "__main__":
    main()
