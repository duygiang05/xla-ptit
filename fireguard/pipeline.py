
"""FireGuard video detection pipeline.

TV1: Color Detection
TV2: Frame Difference / MOG2
Fusion: Color + Motion
TV3: Morphology + Region Extraction
TV4: Tracking + Temporal Decision

FIRE:
    Frame Difference + Fusion + Temporal Vote

SMOKE:
    Frame Difference / MOG2
    + Fusion
    + Optional Shape + Edge verification
    + Temporal Vote / Hybrid

FIRE and SMOKE use separate TemporalFilter instances.
"""

import argparse
import csv
from pathlib import Path

import cv2

from .config import Config
from .color import color_candidates
from .motion import MotionDetector, MOG2MotionDetector
from .fusion import fuse_masks
from .shape import extract_regions
from .temporal import TemporalFilter


FUSION_MODES = (
    "and",
    "and_dilate",
    "color_only",
    "region_motion",
)

SMOKE_VERIFY_MODES = (
    "off",
    "shape_edge",
)


# ============================================================
# SMOKE: Shape + Edge Verification
# ============================================================

def verify_smoke_regions(
    regions,
    smoke_mask,
    frame_bgr,
    max_aspect_ratio=1.10,
    max_edge_density=0.02,
):
    """Filter SMOKE candidates using shape and edge features.

    The candidate-region criteria were derived from
    exploratory measurements on demo2.mp4 and dog.mp4.

    Aspect ratio = bounding box width / height.

    Edge density is calculated only on foreground
    pixels of the smoke mask inside each region bbox.

    This is experimental verification, not a trained
    classifier and not a calibrated probability.
    """

    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    verified = []

    height, width = smoke_mask.shape

    for region in regions:

        # TV3 already provides aspect ratio.
        if region.aspect_ratio > max_aspect_ratio:
            continue

        x, y, w, h = region.bbox

        if w <= 0 or h <= 0:
            continue

        # Clamp bbox to valid image coordinates.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x1 >= x2 or y1 >= y2:
            continue

        roi_mask = (
            smoke_mask[y1:y2, x1:x2] > 0
        )

        if not roi_mask.any():
            continue

        roi_edges = edges[
            y1:y2,
            x1:x2,
        ]

        # Match the earlier feature-analysis method.
        edge_density = float(
            (roi_edges[roi_mask] > 0).mean()
        )

        if edge_density > max_edge_density:
            continue

        verified.append(region)

    return verified


# ============================================================
# Main Pipeline
# ============================================================

def run_video(
    input_path: str,
    output_path: str,
    csv_path: str | None = None,
    cfg: Config | None = None,
    *,
    fire_fusion: str = "and",
    smoke_fusion: str = "and",
    motion_dilate: int = 5,
    smoke_motion: str = "frame_diff",
    smoke_motion_dilate: int | None = None,
    smoke_region_close: int = 15,
    smoke_region_motion_ratio: float = 0.015,
    smoke_region_min_motion_pixels: int = 12,
    smoke_region_min_area: int = 130,
    smoke_temporal_mode: str = "vote",
    smoke_verify: str = "off",
    smoke_max_aspect_ratio: float = 1.10,
    smoke_max_edge_density: float = 0.02,
) -> dict:
    """Run integrated FIRE and SMOKE detection.

    FIRE always uses Frame Difference and Temporal Vote.

    SMOKE may use MOG2, region-level fusion,
    shape/edge verification, and temporal voting.

    New verification is disabled by default
    to preserve existing behavior.
    """

    cfg = cfg or Config()

    # --------------------------------------------------------
    # Validate options
    # --------------------------------------------------------

    if smoke_motion not in ("frame_diff", "mog2"):
        raise ValueError(
            "smoke_motion must be 'frame_diff' or 'mog2'"
        )

    if smoke_temporal_mode not in ("vote", "hybrid"):
        raise ValueError(
            "smoke_temporal_mode must be 'vote' or 'hybrid'"
        )

    if smoke_verify not in SMOKE_VERIFY_MODES:
        raise ValueError(
            f"smoke_verify must be one of {SMOKE_VERIFY_MODES}"
        )

    if smoke_max_aspect_ratio <= 0:
        raise ValueError(
            "smoke_max_aspect_ratio must be positive"
        )

    if not 0 <= smoke_max_edge_density <= 1:
        raise ValueError(
            "smoke_max_edge_density must be between 0 and 1"
        )

    if smoke_motion_dilate is None:
        smoke_motion_dilate = motion_dilate

    # --------------------------------------------------------
    # Open input video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(
            f"Cannot read video: {input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_path = Path(output_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cfg.width, cfg.height),
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError(
            f"Cannot write output video: {output_path}"
        )

    # --------------------------------------------------------
    # TV2: Motion Detectors
    # --------------------------------------------------------

    fire_motion_detector = MotionDetector(cfg)

    smoke_motion_detector = (
        MOG2MotionDetector(
            history=200,
            var_threshold=16,
        )
        if smoke_motion == "mog2"
        else None
    )

    # --------------------------------------------------------
    # TV4: Independent Temporal Filters
    # --------------------------------------------------------

    fire_temporal = TemporalFilter(
        cfg,
        decision_mode="vote",
    )

    smoke_temporal = TemporalFilter(
        cfg,
        decision_mode=smoke_temporal_mode,
    )

    frame_no = 0
    events = []

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.resize(
                frame,
                (cfg.width, cfg.height),
                interpolation=cv2.INTER_AREA,
            )

            # ====================================================
            # TV1: Color Detection
            # ====================================================

            colors = color_candidates(frame, cfg)

            # ====================================================
            # TV2: Motion Detection
            # ====================================================

            fire_movement = fire_motion_detector.apply(
                frame
            )

            if smoke_motion_detector is not None:
                smoke_movement = smoke_motion_detector.apply(
                    frame
                )
            else:
                smoke_movement = fire_movement

            # ====================================================
            # Fusion: FIRE
            # ====================================================

            fire_mask = fuse_masks(
                colors["fire"],
                fire_movement,
                mode=fire_fusion,
                motion_dilate=motion_dilate,
            )

            # ====================================================
            # Fusion: SMOKE
            # ====================================================

            smoke_mask = fuse_masks(
                colors["smoke"],
                smoke_movement,
                mode=smoke_fusion,
                motion_dilate=smoke_motion_dilate,
                region_close=smoke_region_close,
                region_motion_ratio=smoke_region_motion_ratio,
                region_min_motion_pixels=smoke_region_min_motion_pixels,
                region_min_area=smoke_region_min_area,
            )

            # ====================================================
            # TV3: Morphology + Region Extraction
            # ====================================================

            fire_regions = extract_regions(
                fire_mask,
                "fire",
                cfg,
            )

            smoke_regions = extract_regions(
                smoke_mask,
                "smoke",
                cfg,
            )

            # ====================================================
            # SMOKE: Experimental Shape + Edge Verification
            # ====================================================

            if smoke_verify == "shape_edge":
                smoke_regions = verify_smoke_regions(
                    regions=smoke_regions,
                    smoke_mask=smoke_mask,
                    frame_bgr=frame,
                    max_aspect_ratio=smoke_max_aspect_ratio,
                    max_edge_density=smoke_max_edge_density,
                )

            # ====================================================
            # TV4: Independent Tracking + Temporal Decisions
            # ====================================================

            confirmed_fire = fire_temporal.update(
                fire_regions,
                frame_no,
            )

            confirmed_smoke = smoke_temporal.update(
                smoke_regions,
                frame_no,
            )

            confirmed = (
                confirmed_fire + confirmed_smoke
            )

            # ====================================================
            # Draw Detection Results
            # ====================================================

            annotated = frame.copy()

            timestamp = frame_no / fps

            for obj in confirmed:
                x, y, w, h = obj.bbox

                color = (
                    (0, 0, 255)
                    if obj.kind == "fire"
                    else (255, 200, 0)
                )

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2,
                )

                cv2.putText(
                    annotated,
                    obj.kind.upper(),
                    (x, max(y - 7, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                events.append([
                    frame_no,
                    round(timestamp, 3),
                    obj.kind,
                    x,
                    y,
                    w,
                    h,
                    obj.score,
                ])

            writer.write(annotated)

            frame_no += 1

    finally:
        cap.release()
        writer.release()

    # --------------------------------------------------------
    # CSV Export
    # --------------------------------------------------------

    if csv_path:
        csv_file = Path(csv_path)

        csv_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with csv_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow([
                "frame",
                "timestamp_sec",
                "kind",
                "x",
                "y",
                "w",
                "h",
                "score",
            ])

            csv_writer.writerows(events)

    return {
        "frames": frame_no,
        "event_rows": len(events),
        "output_video": str(out_path),
    }


# ============================================================
# Command-line Interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FireGuard FIRE/SMOKE detection"
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="outputs/videos/out.mp4",
    )

    parser.add_argument(
        "--csv",
        default="outputs/events/events.csv",
    )

    parser.add_argument(
        "--fire-fusion",
        choices=FUSION_MODES,
        default="and",
    )

    parser.add_argument(
        "--smoke-fusion",
        choices=FUSION_MODES,
        default="and",
    )

    parser.add_argument(
        "--motion-dilate",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--smoke-motion",
        choices=("frame_diff", "mog2"),
        default="frame_diff",
    )

    parser.add_argument(
        "--smoke-motion-dilate",
        type=int,
        default=None,
    )

    # --------------------------------------------------------
    # SMOKE Region-Level Fusion
    # --------------------------------------------------------

    parser.add_argument(
        "--smoke-region-close",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--smoke-region-motion-ratio",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--smoke-region-min-motion-pixels",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--smoke-region-min-area",
        type=int,
        default=130,
    )

    # --------------------------------------------------------
    # SMOKE Temporal Decision
    # --------------------------------------------------------

    parser.add_argument(
        "--smoke-temporal-mode",
        choices=("vote", "hybrid"),
        default="vote",
    )

    # --------------------------------------------------------
    # SMOKE Shape + Edge Verification
    # --------------------------------------------------------

    parser.add_argument(
        "--smoke-verify",
        choices=SMOKE_VERIFY_MODES,
        default="off",
        help="Optional SMOKE candidate verification",
    )

    parser.add_argument(
        "--smoke-max-aspect-ratio",
        type=float,
        default=1.10,
    )

    parser.add_argument(
        "--smoke-max-edge-density",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    result = run_video(
        args.input,
        args.output,
        args.csv,
        fire_fusion=args.fire_fusion,
        smoke_fusion=args.smoke_fusion,
        motion_dilate=args.motion_dilate,
        smoke_motion=args.smoke_motion,
        smoke_motion_dilate=args.smoke_motion_dilate,
        smoke_region_close=args.smoke_region_close,
        smoke_region_motion_ratio=args.smoke_region_motion_ratio,
        smoke_region_min_motion_pixels=(
            args.smoke_region_min_motion_pixels
        ),
        smoke_region_min_area=args.smoke_region_min_area,
        smoke_temporal_mode=args.smoke_temporal_mode,
        smoke_verify=args.smoke_verify,
        smoke_max_aspect_ratio=args.smoke_max_aspect_ratio,
        smoke_max_edge_density=args.smoke_max_edge_density,
    )

    print(result)
