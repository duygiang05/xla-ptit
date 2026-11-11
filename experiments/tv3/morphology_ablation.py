"""TV3 – Thực nghiệm hình thái học, trích vùng và đặc trưng hình dạng.

Chạy từ thư mục gốc repo:
    python experiments/tv3/morphology_ablation.py                 # chỉ dữ liệu tổng hợp
    python experiments/tv3/morphology_ablation.py --video data/development/x.mp4 --max-frames 300

Kết quả ghi vào experiments/tv3/results/:
  1. ablation_synthetic.csv + ablation_grid.png : so sánh không/open/close/open+close
     với các kích thước kernel trên mask có ground truth (IoU pixel, số vùng, vùng sai).
  2. contours_vs_cc.csv : findContours so với connectedComponentsWithStats (số vùng, thời gian).
  3. tracks_synthetic.csv + tracks_synthetic.png : chuỗi 30 frame giả lập gồm "lửa" dao động
     và "đèn" đứng yên -> đặc trưng biến đổi hình dạng theo thời gian của từng track.
  4. (nếu có --video) video_stats.csv, video_tracks.csv, frames/*.png : mask thật từ TV1+TV2.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fireguard.config import Config  # noqa: E402
from fireguard.shape import (RegionTracker, apply_morphology, combine_masks, draw_regions,  # noqa: E402
                             extract_regions, extract_regions_cc)

OUT = ROOT / "experiments" / "tv3" / "results"
OPS = {"none": (), "open": ("open",), "close": ("close",),
       "open+close": ("open", "close"), "close+open": ("close", "open")}
KERNELS = (3, 5, 7, 9)


# ---------------------------------------------------------------- helpers
def flame_polygon(cx, cy, r_out, r_in, n=7, phase=0.0, rng=None):
    """Đa giác 'ngọn lửa': thân tròn phía dưới, các lưỡi lửa nhọn kéo dài lên trên."""
    pts = []
    for i in range(2 * n):
        ang = math.pi * i / n + phase
        tip = i % 2 == 0 and math.sin(ang) < 0.2  # trục y ảnh hướng xuống: sin<0 là nửa trên
        rad = r_out if tip else r_in
        if rng is not None:
            rad *= rng.uniform(0.85, 1.15)
        stretch = 1.5 if tip else 1.0
        pts.append((cx + rad * math.cos(ang), cy + rad * stretch * math.sin(ang)))
    return np.array(pts, dtype=np.int32)


def synthetic_masks(seed=0, h=360, w=640):
    """Ground truth gồm 3 vùng (2 'lửa' răng cưa + 1 'khói' tròn mờ) và bản nhiễu của nó."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), np.uint8)
    cv2.fillPoly(gt, [flame_polygon(150, 220, 80, 50, rng=rng)], 255)
    cv2.fillPoly(gt, [flame_polygon(340, 250, 55, 35, n=6, rng=rng)], 255)
    cv2.ellipse(gt, (510, 150), (80, 55), 15, 0, 360, 255, -1)
    noisy = gt.copy()
    # salt: chấm nhiễu nhỏ 1-3 px ngoài vùng (do nén video, lá cây, phản chiếu...)
    for _ in range(400):
        y, x, s = rng.integers(0, h), rng.integers(0, w), rng.integers(1, 4)
        noisy[y:y + s, x:x + s] = 255
    # pepper: lỗ thủng bên trong vùng (pixel không qua ngưỡng màu/chuyển động)
    ys, xs = np.nonzero(gt)
    idx = rng.choice(len(ys), size=len(ys) // 15, replace=False)
    noisy[ys[idx], xs[idx]] = 0
    # vết đứt mảnh cắt ngang vùng khói -> tách thành 2 mảnh nếu không closing
    cv2.line(noisy, (440, 150), (590, 160), 0, 2)
    return gt, noisy, 3


def pixel_iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def labeled(img, text):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- 1. ablation
def run_ablation(cfg):
    print("[1] Morphology ablation trên mask tổng hợp")
    gt, noisy, n_true = synthetic_masks()
    rows, best = [], None
    for name, ops in OPS.items():
        for k in (KERNELS if ops else (0,)):
            t0 = time.perf_counter()
            for _ in range(20):
                cleaned = apply_morphology(noisy, ops, k) if ops else noisy.copy()
            ms = (time.perf_counter() - t0) / 20 * 1000
            regions = extract_regions(cleaned, "fire", cfg, clean=False)
            matched = sum(1 for r in regions if gt[int(r.centroid[1]), int(r.centroid[0])] > 0)
            iou = pixel_iou(cleaned, gt)
            # điểm tổng hợp: IoU pixel trừ phạt khi số vùng lệch số vùng thật (vỡ vùng / vùng thừa)
            score = iou - 0.05 * abs(len(regions) - n_true)
            row = {"ops": name, "kernel": k, "pixel_iou": round(iou, 4),
                   "n_regions": len(regions), "n_true": n_true, "false_regions": len(regions) - matched,
                   "score": round(score, 4), "morph_ms": round(ms, 3)}
            rows.append(row)
            if best is None or score > best[0]["score"]:
                best = (row, cleaned, regions)
    write_csv(OUT / "ablation_synthetic.csv", rows)
    for r in rows:
        print(f"    {r['ops']:<11} k={r['kernel']}  IoU={r['pixel_iou']:.3f}  vùng={r['n_regions']:>3} "
              f"(sai {r['false_regions']:>3})  score={r['score']:+.3f}  {r['morph_ms']:.2f} ms")

    # lưới minh họa: GT | noisy -> contour | best cleaned -> bbox
    raw_regions = extract_regions(noisy, "fire", Config(min_region_area=1), clean=False)
    top = np.hstack([labeled(gt, "Ground truth"), labeled(noisy, "Mask nhieu (ung vien tho)")])
    contour_img = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(noisy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 1)
    row, cleaned, regions = best
    bot = np.hstack([
        labeled(contour_img, f"Khong morphology: {len(raw_regions)} contour"),
        labeled(draw_regions(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR), regions),
                f"{row['ops']} k={row['kernel']}: {len(regions)} vung, IoU={row['pixel_iou']:.2f}"),
    ])
    cv2.imwrite(str(OUT / "ablation_grid.png"), np.vstack([top, bot]))
    print(f"  -> experiments/tv3/results/ablation_grid.png  (tốt nhất: {row['ops']} k={row['kernel']})")


# ---------------------------------------------------------------- 2. contours vs CC
def run_contours_vs_cc(cfg):
    print("[2] findContours vs connectedComponentsWithStats")
    _, noisy, _ = synthetic_masks(seed=1)
    rows = []
    for clean in (False, True):
        for name, fn in (("findContours", extract_regions), ("connectedComponents", extract_regions_cc)):
            t0 = time.perf_counter()
            for _ in range(20):
                regions = fn(noisy, "fire", cfg, clean=clean)
            ms = (time.perf_counter() - t0) / 20 * 1000
            rows.append({"method": name, "morphology": clean, "n_regions": len(regions),
                         "total_area": round(sum(r.area for r in regions), 1), "ms_per_frame": round(ms, 3)})
            print(f"    {name:<20} morph={clean!s:<5} vùng={len(regions):>3} "
                  f"tổng S={rows[-1]['total_area']:>9}  {ms:.2f} ms")
    write_csv(OUT / "contours_vs_cc.csv", rows)


# ---------------------------------------------------------------- 3. tracks
def run_tracks(cfg, n_frames=30):
    print("[3] Theo dõi vùng + biến đổi hình dạng trên chuỗi giả lập")
    rng = np.random.default_rng(2)
    tracker = RegionTracker(cfg, history=n_frames)
    labels, rows = {}, []
    for i in range(n_frames):
        mask = np.zeros((360, 640), np.uint8)
        # "lửa": kích thước + biên thay đổi mỗi frame, trôi nhẹ lên trên
        r_out = 70 + 20 * math.sin(i * 0.9) + rng.uniform(-8, 8)
        cv2.fillPoly(mask, [flame_polygon(200, 230 - i * 0.8, r_out, r_out * 0.45, n=10,
                                          phase=rng.uniform(0, 0.3), rng=rng)], 255)
        # "đèn": cùng màu nhưng đứng yên, gần như không đổi hình dạng
        cv2.circle(mask, (480, 160), 40 + int(rng.integers(0, 2)), 255, -1)
        regions = extract_regions(mask, "fire", cfg)
        for tid, r in tracker.update(regions, i):
            labels.setdefault(tid, "flame (dao dong)" if r.centroid[0] < 340 else "lamp (dung yen)")
        if i == n_frames - 1:
            cv2.imwrite(str(OUT / "tracks_synthetic.png"), draw_regions(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), regions))
    for t in tracker.tracks:
        stats = {k: round(v, 4) for k, v in t.shape_change().items()}
        rows.append({"track_id": t.track_id, "object": labels[t.track_id], **stats})
        print(f"    track {t.track_id} {labels[t.track_id]:<17} area_cv={stats['area_cv']:.3f} "
              f"circ_std={stats['circularity_std']:.3f} shift={stats['centroid_shift']:.2f}px "
              f"IoU={stats['bbox_iou_mean']:.3f}")
    write_csv(OUT / "tracks_synthetic.csv", rows)


# ---------------------------------------------------------------- 4. real video
def run_video(cfg, path, max_frames, save_every, combine_mode, motion_dilate):
    from fireguard.color import color_candidates
    from fireguard.motion import MotionDetector

    print(f"[4] Video thật: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {path}")
    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    motion, trackers = MotionDetector(cfg), {"fire": RegionTracker(cfg), "smoke": RegionTracker(cfg)}
    stats, all_tracks, i = [], {}, 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        colors, mov = color_candidates(frame, cfg), motion.apply(frame)
        for kind in ("fire", "smoke"):
            combined = combine_masks(colors[kind], mov, combine_mode, motion_dilate)
            raw = extract_regions(combined, kind, cfg, clean=False)
            clean = extract_regions(combined, kind, cfg, clean=True)
            for tid, r in trackers[kind].update(clean, i):
                all_tracks[(kind, tid)] = trackers[kind].get(tid)
            stats.append({"frame": i, "kind": kind, "mask_px": int(np.count_nonzero(combined)),
                          "regions_raw": len(raw), "regions_morph": len(clean),
                          "area_morph": round(sum(r.area for r in clean), 1)})
            if save_every and i % save_every == 0 and np.count_nonzero(combined):
                cleaned = apply_morphology(combined, ("open", "close"), cfg.morph_kernel)
                small = lambda im: cv2.resize(im, (cfg.width // 2, cfg.height // 2))  # noqa: E731
                grid = np.vstack([
                    np.hstack([small(labeled(frame, "Frame goc")), small(labeled(combined, f"{kind}: mau {combine_mode} motion"))]),
                    np.hstack([small(labeled(cleaned, "Sau open+close")),
                               small(labeled(draw_regions(frame, clean), f"{len(clean)} vung (raw {len(raw)})"))]),
                ])
                cv2.imwrite(str(frames_dir / f"{kind}_{i:05d}.png"), grid)
        i += 1
    cap.release()
    if not stats:
        print("  Video không có frame nào.")
        return
    write_csv(OUT / "video_stats.csv", stats)
    track_rows = [{"kind": k, "track_id": tid, **{a: round(b, 4) for a, b in t.shape_change().items()}}
                  for (k, tid), t in all_tracks.items() if t is not None and t.length >= 3]
    if track_rows:
        write_csv(OUT / "video_tracks.csv", track_rows)
    for kind in ("fire", "smoke"):
        s = [r for r in stats if r["kind"] == kind]
        print(f"    {kind}: TB vùng/frame không morph={np.mean([r['regions_raw'] for r in s]):.2f}, "
              f"có morph={np.mean([r['regions_morph'] for r in s]):.2f} ({len(s)} frame)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video thật để chạy phần [4]")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=30, help="lưu ảnh debug mỗi N frame (0 = tắt)")
    ap.add_argument("--combine", choices=("and", "or", "color"), default="and")
    ap.add_argument("--motion-dilate", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=None, help="ghi đè Config.min_region_area")
    ap.add_argument("--kernel", type=int, default=None, help="ghi đè Config.morph_kernel")
    args = ap.parse_args()

    base = Config()
    cfg = Config(**{**base.__dict__,
                    "min_region_area": args.min_area if args.min_area is not None else base.min_region_area,
                    "morph_kernel": args.kernel if args.kernel is not None else base.morph_kernel})
    OUT.mkdir(parents=True, exist_ok=True)
    run_ablation(cfg)
    run_contours_vs_cc(cfg)
    run_tracks(cfg)
    if args.video:
        run_video(cfg, args.video, args.max_frames, args.save_every, args.combine, args.motion_dilate)


if __name__ == "__main__":
    main()
