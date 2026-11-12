"""TV1 – Thí nghiệm 4: CHỈ MÀU so với MÀU + CHUYỂN ĐỘNG (color ablation).

Câu hỏi: mask màu của TV1 bắt nhầm đèn đỏ, đèn sodium, tường, mây... Khi ghép với mask
chuyển động thì các báo nhầm đó có bị loại không, và lửa/khói thật còn giữ được bao nhiêu?

Chạy từ thư mục gốc repo:
    python experiments/tv1/color_vs_motion.py

Chỉ GỌI code của TV2 (MotionDetector, MOG2MotionDetector) và TV3 (combine_masks,
extract_regions) như hộp đen với tham số mặc định – không sửa, không chỉnh tham số của họ.

Dữ liệu: chuỗi TỔNG HỢP 60 frame 640x360, camera cố định, nhiễu cảm biến mỗi frame.
  - fire      : ngọn lửa lập lòe (hình dạng đổi mỗi frame) – màu lửa, CÓ chuyển động
  - smoke     : cột khói bán trong suốt bay lên            – màu khói, CÓ chuyển động
  - traffic_red, sodium : đèn đỏ / đèn cam đứng yên         – màu lửa, KHÔNG chuyển động
  - concrete, cloud     : tường xám / mây đứng yên          – màu khói, KHÔNG chuyển động
  - red_car   : xe đỏ chạy ngang                            – màu lửa, CÓ chuyển động (!)
Kết quả ghi vào experiments/tv1/results/:
  color_vs_motion.csv, color_vs_motion_grid.png
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fireguard.color import color_candidates  # noqa: E402
from fireguard.config import Config  # noqa: E402
from fireguard.motion import MOG2MotionDetector, MotionDetector  # noqa: E402
from fireguard.shape import combine_masks, extract_regions  # noqa: E402
from _utils import imwrite_unicode, label  # noqa: E402
from compare_rgb_hsv import flame_layers  # noqa: E402

OUT = ROOT / "experiments" / "tv1" / "results"
H, W = 360, 640
N_FRAMES, WARMUP = 60, 10
METHODS = ("color_only", "and_diff", "and_diff_dilate15", "and_mog2")
DISTRACTORS = {"fire": ("traffic_red", "sodium", "red_car"), "smoke": ("concrete", "cloud")}
SAMPLE_FRAME = 35
# Khói nhạt có độ sáng gần bằng trời (khó thấy với ảnh xám); khói đen tương phản cao.
# Kịch bản: (màu khói, có vân xoáy). "smooth" là đối chứng cho thấy ảnh hưởng của vân.
SCENARIOS = {"light_smooth": ((175, 173, 170), False), "light_turbulent": ((175, 173, 170), True),
             "dark_turbulent": ((85, 88, 92), True)}


def static_background():
    """Nền đứng yên + mask của các vật đứng yên."""
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (205, 165, 115)                                   # trời
    img[230:] = (128, 128, 126)                                # tường/nền bê tông
    m = {}
    m["concrete"] = np.zeros((H, W), bool); m["concrete"][230:] = True
    cloud = np.zeros((H, W), np.uint8)
    cv2.ellipse(cloud, (560, 55), (70, 28), 0, 0, 360, 255, -1)
    img[cloud > 0] = (215, 212, 208); m["cloud"] = cloud > 0
    cv2.rectangle(img, (300, 70), (340, 170), (40, 40, 40), -1)
    light = np.zeros((H, W), np.uint8); cv2.circle(light, (320, 95), 15, 255, -1)
    img[light > 0] = (30, 30, 240); m["traffic_red"] = light > 0
    lamp = np.zeros((H, W), np.uint8); cv2.circle(lamp, (60, 70), 28, 255, -1)
    img[lamp > 0] = (40, 150, 255); m["sodium"] = lamp > 0
    return img, m


RISE = 4  # khói bay lên RISE pixel mỗi frame
# Trường "vân xoáy" mịn (nhiễu đã blur), cao hơn ảnh để có thể trượt lên theo thời gian.
_TURB = cv2.GaussianBlur(np.random.default_rng(42).random((H + RISE * N_FRAMES, W)).astype(np.float32),
                         (0, 0), 5)
_TURB = (_TURB - _TURB.mean()) / _TURB.std()  # chuẩn hóa: trung bình 0, độ lệch chuẩn 1


def smoke_alpha(t, turbulent=True):
    """Cột khói bay lên: các đốm Gauss trôi dần lên trên, nở rộng và loãng dần.

    turbulent=True: nhân thêm vân xoáy trôi lên cùng khói. Khói thật có các cuộn đặc/loãng;
    thiếu vân này, khói tổng hợp trơn láng tới mức frame difference gần như không thấy gì.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    alpha = np.zeros((H, W), np.float32)
    for i in range(10):
        p = ((t * RISE + i * 22) % 220) / 220                  # vị trí 0 (chân) -> 1 (đỉnh)
        cx = 470 + 20 * np.sin(6 * p + 0.3 * i)
        cy = 225 - p * 210
        sigma = 18 + 30 * p
        alpha += (1 - 0.7 * p) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    if turbulent:
        # Lấy lát cắt dịch dần xuống trong _TURB -> nội dung vân trôi LÊN RISE px mỗi frame
        alpha *= np.clip(1 + 0.6 * _TURB[t * RISE:t * RISE + H], 0.2, 1.8)  # cuộn đặc/loãng rõ
    return np.clip(alpha * 0.55, 0, 0.9)


def make_frame(t, bg, static, rng, smoke_color=(175, 173, 170), turbulent=True):
    img = bg.copy()
    gt = {k: v.copy() for k, v in static.items()}

    # Xe đỏ chạy từ phải sang trái trên nền bê tông, ra khỏi khung từ frame ~50
    car = np.zeros((H, W), np.uint8)
    x = 360 - 9 * t                                            # x âm: OpenCV tự cắt phần ngoài ảnh
    cv2.rectangle(car, (x, 290), (x + 90, 325), 255, -1)
    img[car > 0] = (35, 30, 195)
    gt["red_car"] = car > 0

    # Ngọn lửa lập lòe: kích thước và vị trí dao động ngẫu nhiên mỗi frame
    fire = np.zeros((H, W), np.uint8)
    scale = 0.85 + 0.25 * rng.random()
    cx = 170 + rng.integers(-4, 5)
    for col, pts in flame_layers(cx, 300, scale):
        cv2.fillPoly(img, [pts], col)
        cv2.fillPoly(fire, [pts], 255)
    gt["fire"] = fire > 0

    a = smoke_alpha(t, turbulent)
    img = (1 - a[..., None]) * img + a[..., None] * np.array(smoke_color, np.float32)
    gt["smoke"] = a >= 0.3
    for k in ("concrete", "cloud", "traffic_red", "sodium", "red_car"):
        gt[k] &= a < 0.1                                       # phần bị khói che không tính là nhiễu
    for k in ("concrete", "cloud", "traffic_red", "sodium"):
        gt[k] &= ~gt["fire"] & ~gt["red_car"]

    img = cv2.GaussianBlur(img, (5, 5), 0) + rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), gt


def region_stats(mask, kind, gt, cfg):
    """Dùng extract_regions của TV3: có tìm thấy vùng thật không, bao nhiêu vùng báo nhầm.

    Vùng tính là ĐÚNG khi >= 30% diện tích bbox là lửa/khói thật (chỉ "chạm" là chưa đủ:
    dải tường rộng có bbox phủ lên chân cột khói sẽ bị tính sai là tìm thấy khói).
    """
    hit, false = 0, 0
    for r in extract_regions(mask, kind, cfg):
        x, y, w, h = r.bbox
        if gt[kind][y:y + h, x:x + w].mean() >= 0.3:
            hit = 1
        else:
            false += 1
    return hit, false


def run_scenario(scenario, smoke_color, turbulent, cfg):
    rng = np.random.default_rng(0)
    bg, static = static_background()
    diff_det, mog2_det = MotionDetector(cfg), MOG2MotionDetector()

    acc = {(k, m): {"recall": [], "hit": [], "false_regions": [], **{d: [] for d in DISTRACTORS[k]}}
           for k in ("fire", "smoke") for m in METHODS}
    sample = None
    for t in range(N_FRAMES):
        frame, gt = make_frame(t, bg, static, rng, smoke_color, turbulent)
        colors = color_candidates(frame, cfg)
        motion = {"diff": diff_det.apply(frame), "mog2": mog2_det.apply(frame)}
        masks = {}
        for kind in ("fire", "smoke"):
            c = colors[kind]
            masks[(kind, "color_only")] = combine_masks(c, motion["diff"], "color")
            masks[(kind, "and_diff")] = combine_masks(c, motion["diff"], "and")
            masks[(kind, "and_diff_dilate15")] = combine_masks(c, motion["diff"], "and", motion_dilate=15)
            masks[(kind, "and_mog2")] = combine_masks(c, motion["mog2"], "and")
        if t == SAMPLE_FRAME:
            sample = (frame, motion, masks)
        if t < WARMUP:
            continue
        for (kind, method), mk in masks.items():
            on, a = mk > 0, acc[(kind, method)]
            a["recall"].append(np.count_nonzero(on & gt[kind]) / max(1, np.count_nonzero(gt[kind])))
            for d in DISTRACTORS[kind]:
                if gt[d].any():
                    a[d].append(np.count_nonzero(on & gt[d]) / np.count_nonzero(gt[d]))
            hit, false = region_stats(mk, kind, gt, cfg)
            a["hit"].append(hit)
            a["false_regions"].append(false)

    rows = []
    for (kind, method), a in acc.items():
        row = {"scenario": scenario, "kind": kind, "method": method,
               "pixel_recall": np.mean(a["recall"]),
               "frames_found": np.mean(a["hit"]),
               "false_regions_per_frame": np.mean(a["false_regions"])}
        for d in ("traffic_red", "sodium", "red_car", "concrete", "cloud"):
            row[f"fp_{d}"] = np.mean(a[d]) if d in a and a[d] else ""
        rows.append(row)
    return rows, sample


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    rows, samples = [], {}
    for scenario, (smoke_color, turbulent) in SCENARIOS.items():
        r, samples[scenario] = run_scenario(scenario, smoke_color, turbulent, cfg)
        rows += r

    with (OUT / "color_vs_motion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    frame, motion, masks = samples["light_turbulent"]
    dframe, dmotion, dmasks = samples["dark_turbulent"]
    top = [label(frame, f"input t={SAMPLE_FRAME}"), label(motion["diff"], "TV2 frame diff"),
           label(motion["mog2"], "TV2 MOG2"), np.zeros_like(frame)]
    fire_row = [label(masks[("fire", m)], f"fire | {m}") for m in METHODS]
    smoke_row = [label(masks[("smoke", m)], f"smoke | {m}") for m in METHODS]
    dark_top = [label(dframe, "dark smoke input"), label(dmotion["diff"], "frame diff (dark smoke)"),
                label(dmotion["mog2"], "MOG2 (dark smoke)"), np.zeros_like(frame)]
    dark_row = [label(dmasks[("smoke", m)], f"dark smoke | {m}") for m in METHODS]
    grid = np.vstack([np.hstack(top), np.hstack(fire_row), np.hstack(smoke_row),
                      np.hstack(dark_top), np.hstack(dark_row)])
    imwrite_unicode(OUT / "color_vs_motion_grid.png",
                    cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    cols = list(rows[0])
    print(" | ".join(c.replace("fp_", "") for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    print(f"\nĐã lưu kết quả vào {OUT}")


if __name__ == "__main__":
    main()
