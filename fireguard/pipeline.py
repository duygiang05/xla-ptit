"""Pipeline FireGuard: TV1 color + TV2 motion -> Fusion -> TV3 -> TV4.

Default fusion remains the old hard AND for backwards compatibility.
Select per-branch modes via CLI or run_video keyword arguments.
"""
import argparse
import csv
from pathlib import Path
import cv2
from .config import Config
from .color import color_candidates
from .motion import MotionDetector
from .fusion import fuse_masks
from .shape import extract_regions
from .temporal import TemporalFilter


def run_video(input_path: str, output_path: str, csv_path: str | None = None,
              cfg: Config | None = None, *, fire_fusion: str = "and",
              smoke_fusion: str = "and", motion_dilate: int = 5) -> dict:
    """Process video with independently selectable fusion modes for FIRE/SMOKE.

    The existing positional parameters are unchanged; app.py works unchanged.
    """
    cfg = cfg or Config()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot read video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (cfg.width, cfg.height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot write output video: {output_path}")
    motion = MotionDetector(cfg)
    temporal = TemporalFilter(cfg)
    frame_no = 0
    events = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
            colors = color_candidates(frame, cfg)
            movement = motion.apply(frame)

            # Fusion nhận kết quả TV1 & TV2; TV3 CHỈ nhận mask sau Fusion.
            fire_mask = fuse_masks(colors["fire"], movement,
                                   mode=fire_fusion, motion_dilate=motion_dilate)
            smoke_mask = fuse_masks(colors["smoke"], movement,
                                    mode=smoke_fusion, motion_dilate=motion_dilate)
            regions = (extract_regions(fire_mask, "fire", cfg)
                       + extract_regions(smoke_mask, "smoke", cfg))
            confirmed = temporal.update(regions, frame_no)

            annotated = frame.copy()
            timestamp = frame_no / fps
            for obj in confirmed:
                x, y, w, h = obj.bbox
                col = (0, 0, 255) if obj.kind == "fire" else (255, 200, 0)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), col, 2)
                cv2.putText(annotated, obj.kind.upper(), (x, max(y - 7, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                events.append([frame_no, round(timestamp, 3), obj.kind,
                               x, y, w, h, obj.score])
            writer.write(annotated)
            frame_no += 1
    finally:
        cap.release()
        writer.release()
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame", "timestamp_sec", "kind", "x", "y", "w", "h", "score"])
            csv_writer.writerows(events)
    return {"frames": frame_no, "event_rows": len(events),
            "output_video": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/videos/out.mp4")
    parser.add_argument("--csv", default="outputs/events/events.csv")
    parser.add_argument("--fire-fusion", choices=("and", "and_dilate", "color_only"), default="and")
    parser.add_argument("--smoke-fusion", choices=("and", "and_dilate", "color_only"), default="and")
    parser.add_argument("--motion-dilate", type=int, default=5,
                        help="Motion dilation kernel size, e.g. 5 => 5x5; not a radius")
    args = parser.parse_args()
    print(run_video(args.input, args.output, args.csv,
                    fire_fusion=args.fire_fusion,
                    smoke_fusion=args.smoke_fusion,
                    motion_dilate=args.motion_dilate))
