"""So sánh hai luật TV4 trên cùng vùng ứng viên; không đo độ chính xác.

Chạy từ thư mục dự án:
    python experiments/compare_decisions.py --video data/development/ForestFire1.avi
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fireguard.color import color_candidates
from fireguard.config import Config
from fireguard.motion import MotionDetector
from fireguard.shape import extract_regions
from fireguard.temporal import TemporalFilter


def save_csv(path: Path, rows: list[dict]) -> None:
    """Ghi các thống kê theo frame/vùng vào CSV."""
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output-dir", default="experiments/tv4/results")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames phải > 0")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Không mở được video: {video}")

    cfg = Config()
    motion = MotionDetector(cfg)
    filters = {
        "vote": TemporalFilter(cfg, decision_mode="vote"),
        "hybrid": TemporalFilter(cfg, decision_mode="hybrid"),
    }
    frame_rows = []
    region_rows = []
    try:
        for frame_idx in range(args.max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(
                frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA
            )
            masks = color_candidates(frame, cfg)
            movement = motion.apply(frame)
            regions = []
            for kind in ("fire", "smoke"):
                candidate_mask = cv2.bitwise_and(masks[kind], movement)
                regions.extend(extract_regions(candidate_mask, kind, cfg))

            counts = {}
            for mode, temporal_filter in filters.items():
                confirmed = temporal_filter.update(regions, frame_idx)
                counts[mode] = len(confirmed)
                for decision in temporal_filter.last_decisions:
                    region_rows.append({"frame": frame_idx, "mode": mode, **decision})
            frame_rows.append(
                {
                    "frame": frame_idx,
                    "candidate_regions": len(regions),
                    "vote_confirmed": counts["vote"],
                    "hybrid_confirmed": counts["hybrid"],
                }
            )
    finally:
        capture.release()

    if not frame_rows:
        raise SystemExit("Video không có frame đọc được")
    save_csv(output_dir / "comparison_frames.csv", frame_rows)
    if region_rows:
        save_csv(output_dir / "comparison_regions.csv", region_rows)

    print(f"Video: {video.name} | Frames: {len(frame_rows)}")
    for key in ("candidate_regions", "vote_confirmed", "hybrid_confirmed"):
        count = sum(row[key] for row in frame_rows)
        print(f"{key}: {count} lượt vùng/frame")
    print(f"Đã lưu: {output_dir}")
    print("Số vùng giảm không đồng nghĩa số báo sai giảm; cần nhãn để đánh giá.")


if __name__ == "__main__":
    main()
