"""Pipeline video: TV1 màu, TV2 chuyển động, TV4 ghép, TV3 vùng, TV4 thời gian.

FIRE và SMOKE có bộ lọc thời gian riêng; nhánh khói hỗ trợ xử lý theo bối cảnh.
"""

import argparse
import csv
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2

from .config import Config
from .color import color_candidates
from .color.smoke_context import context_smoke_candidates
from .motion import MotionDetector, MOG2MotionDetector
from .motion.flow import dense_flow_in_bounds
from .fusion import fuse_masks
from .shape import extract_regions
from .shape.regions import draw_regions
from .shape.grouping import union_regions
from .shape.context_verify import (
    active_image_bounds,
    extend_sky_plume,
    verify_context_regions,
)
from .temporal import TemporalFilter
from .presets import DEMO_OPTIONS
from .visualization import compose_pipeline_frame, dashboard_size

FUSION_MODES = ("and", "and_dilate", "color_only", "sky_motion", "region_motion", "adaptive_region_motion")

SMOKE_VERIFY_MODES = ("off", "shape_edge")


def verify_smoke_regions(
    regions,
    smoke_mask,
    frame_bgr,
    max_aspect_ratio=1.10,
    max_edge_density=0.02,
):
    """Lọc ứng viên khói theo tỉ lệ hộp và mật độ cạnh trong mask.

    Ngưỡng được thử trên demo2/dog; đây không phải xác suất hay bộ phân loại.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    verified = []

    height, width = smoke_mask.shape

    for region in regions:

        # TV3 đã tính sẵn tỉ lệ rộng/cao của hộp.
        if region.aspect_ratio > max_aspect_ratio:
            continue

        x, y, w, h = region.bbox

        if w <= 0 or h <= 0:
            continue

        # Chặn tọa độ hộp trong phạm vi ảnh.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x1 >= x2 or y1 >= y2:
            continue

        roi_mask = smoke_mask[y1:y2, x1:x2] > 0

        if not roi_mask.any():
            continue

        roi_edges = edges[y1:y2, x1:x2]

        # Chỉ tính mật độ cạnh trên các pixel thuộc mask khói.
        edge_density = float((roi_edges[roi_mask] > 0).mean())

        if edge_density > max_edge_density:
            continue

        verified.append(region)

    return verified


def letterbox_frame(frame, width: int, height: int):
    """Đưa frame vào khung cố định mà vẫn giữ tỉ lệ gốc."""
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    scaled_width = max(1, min(width, round(source_width * scale)))
    scaled_height = max(1, min(height, round(source_height * scale)))
    resized = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    top = (height - scaled_height) // 2
    bottom = height - scaled_height - top
    left = (width - scaled_width) // 2
    right = width - scaled_width - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def run_video(
    input_path: str,
    output_path: str,
    csv_path: str | None = None,
    cfg: Config | None = None,
    *,
    dashboard_path: str | None = None,
    fire_fusion: str = "and",
    smoke_fusion: str | None = None,
    motion_dilate: int = 5,
    smoke_motion: str = "frame_diff",
    smoke_motion_dilate: int | None = None,
    smoke_region_close: int = 15,
    smoke_region_motion_ratio: float = 0.015,
    smoke_region_min_motion_pixels: int = 12,
    smoke_region_min_area: int = 130,
    smoke_region_max_area_ratio: float = 0.50,
    smoke_growth_kernel: int = 5,
    smoke_growth_iterations: int = 8,
    smoke_temporal_mode: str = "vote",
    smoke_verify: str = "off",
    smoke_max_aspect_ratio: float = 1.10,
    smoke_max_edge_density: float = 0.02,
    single_box_per_kind: bool = False,
    smoke_engine: str = "classical",
    smoke_vote_hits: int | None = None,
) -> dict:
    """Phân tích video, ghi MP4 và CSV cho hai nhánh FIRE/SMOKE.

    FIRE dùng sai khác frame và bỏ phiếu thời gian; SMOKE tùy cấu hình dùng
    MOG2, ghép theo vùng và xác minh hình dạng/cạnh. Mặc định của hàm giữ
    hành vi cổ điển để tương thích; app truyền preset khói theo bối cảnh.
    """

    cfg = cfg or Config()

    # Kiểm tra tham số trước khi mở video.

    if smoke_motion not in ("frame_diff", "mog2"):
        raise ValueError("smoke_motion must be 'frame_diff' or 'mog2'")

    if smoke_engine not in ("classical", "opencv_context"):
        raise ValueError("smoke_engine must be 'classical' or 'opencv_context'")
    if smoke_fusion is None:
        smoke_fusion = (
            "adaptive_region_motion" if smoke_engine == "opencv_context" else "and"
        )
    if smoke_vote_hits is not None and not 1 <= smoke_vote_hits <= cfg.vote_window:
        raise ValueError("smoke_vote_hits must be within the vote window")

    if smoke_temporal_mode not in ("vote", "hybrid"):
        raise ValueError("smoke_temporal_mode must be 'vote' or 'hybrid'")

    if smoke_verify not in SMOKE_VERIFY_MODES:
        raise ValueError(f"smoke_verify must be one of {SMOKE_VERIFY_MODES}")

    if smoke_max_aspect_ratio <= 0:
        raise ValueError("smoke_max_aspect_ratio must be positive")

    if not 0 <= smoke_max_edge_density <= 1:
        raise ValueError("smoke_max_edge_density must be between 0 and 1")

    if smoke_motion_dilate is None:
        smoke_motion_dilate = motion_dilate

    # Mở video đầu vào và chuẩn bị MP4 đầu ra.

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot read video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    portrait_grays = deque(maxlen=5)

    out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cfg.width, cfg.height),
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot write output video: {output_path}")

    dashboard_writer = None
    if dashboard_path is not None:
        dashboard_output = Path(dashboard_path)
        if dashboard_output.resolve() == out_path.resolve():
            cap.release()
            writer.release()
            raise ValueError("dashboard_path must differ from output_path")
        dashboard_output.parent.mkdir(parents=True, exist_ok=True)
        dashboard_writer = cv2.VideoWriter(
            str(dashboard_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            dashboard_size(),
        )
        if not dashboard_writer.isOpened():
            cap.release()
            writer.release()
            raise ValueError(f"Cannot write dashboard video: {dashboard_path}")

    # TV2: tạo bộ phát hiện chuyển động riêng cho hai nhánh.

    fire_motion_detector = MotionDetector(cfg)

    smoke_motion_detector = (
        MOG2MotionDetector(history=200, var_threshold=16)
        if smoke_motion == "mog2"
        else None
    )

    # TV4: bộ lọc thời gian FIRE và SMOKE độc lập.

    fire_temporal = TemporalFilter(cfg, decision_mode="vote")

    smoke_temporal = TemporalFilter(
        replace(cfg, vote_hits=smoke_vote_hits) if smoke_vote_hits is not None else cfg,
        decision_mode=smoke_temporal_mode,
    )

    frame_no = 0
    events = []
    last_fire_region = None
    last_fire_frame = -2

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = letterbox_frame(frame, cfg.width, cfg.height)
            active_bounds = (
                active_image_bounds(frame) if smoke_engine == "opencv_context" else None
            )
            portrait_context = (
                active_bounds is not None and active_bounds[2] < 0.6 * cfg.width
            )
            previous_portrait_gray = None
            current_portrait_gray = None
            if portrait_context:
                previous_portrait_gray = (
                    portrait_grays[-4] if len(portrait_grays) >= 4 else None
                )
                current_portrait_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                portrait_grays.append(current_portrait_gray)

            # TV1: tạo ứng viên màu.

            colors = color_candidates(frame, cfg)
            context_sky_scene = False
            context_aerial_tint = False
            context_lower_smoke = None
            if smoke_engine == "opencv_context":
                (
                    colors["smoke"],
                    context_sky_scene,
                    context_aerial_tint,
                    context_lower_smoke,
                ) = context_smoke_candidates(
                    frame, colors["fire"], return_aerial_tint=True, return_lower=True
                )

            # TV2: phát hiện chuyển động.

            fire_movement = fire_motion_detector.apply(frame)

            if smoke_motion_detector is not None:
                smoke_movement = smoke_motion_detector.apply(frame)
            else:
                smoke_movement = fire_movement

            # TV4 Fusion: ghép mask lửa.

            fire_mask = fuse_masks(
                colors["fire"],
                fire_movement,
                mode=fire_fusion,
                motion_dilate=motion_dilate,
            )

            # TV4 Fusion: ghép mask khói.

            smoke_mask = fuse_masks(
                colors["smoke"],
                smoke_movement,
                mode=(
                    (
                        "and_dilate"
                        if context_aerial_tint
                        else (
                            "sky_motion"
                            if context_sky_scene
                            else (
                                "and_dilate"
                                if portrait_context
                                else smoke_fusion
                            )
                        )
                    )
                    if smoke_engine == "opencv_context"
                    else smoke_fusion
                ),
                motion_dilate=(
                    21
                    if context_aerial_tint
                    else (
                        41
                        if context_sky_scene
                        else (13 if portrait_context else smoke_motion_dilate)
                    )
                ),
                region_close=smoke_region_close,
                region_motion_ratio=smoke_region_motion_ratio,
                region_min_motion_pixels=smoke_region_min_motion_pixels,
                region_min_area=smoke_region_min_area,
                region_max_area_ratio=smoke_region_max_area_ratio,
                active_bounds=active_bounds,
                growth_kernel=smoke_growth_kernel,
                growth_iterations=smoke_growth_iterations,
            )

            # TV3: làm sạch mask và trích vùng.

            fire_regions = extract_regions(fire_mask, "fire", cfg)

            smoke_regions = extract_regions(smoke_mask, "smoke", cfg)
            if smoke_engine == "opencv_context":
                portrait_flow = None
                if (
                    portrait_context
                    and not context_sky_scene
                    and smoke_regions
                    and previous_portrait_gray is not None
                ):
                    portrait_flow = dense_flow_in_bounds(
                        previous_portrait_gray, current_portrait_gray, active_bounds
                    )
                smoke_regions = verify_context_regions(
                    smoke_regions,
                    frame,
                    smoke_mask,
                    context_sky_scene,
                    aerial_tint=context_aerial_tint,
                    motion_mask=smoke_movement,
                    fire_regions=fire_regions,
                    portrait_flow=portrait_flow,
                    sky_motion_guided=context_sky_scene,
                )
                if context_sky_scene and cv2.countNonZero(context_lower_smoke):
                    lower_motion = cv2.dilate(
                        smoke_movement,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
                    )
                    lower_mask = cv2.bitwise_and(context_lower_smoke, lower_motion)
                    lower_regions = extract_regions(lower_mask, "smoke", cfg)
                    smoke_regions = extend_sky_plume(
                        smoke_regions, lower_regions, frame
                    )

            # Bộ lọc hình dạng/cạnh khói tùy chọn, đang thử nghiệm.

            if smoke_engine != "opencv_context" and smoke_verify == "shape_edge":
                smoke_regions = verify_smoke_regions(
                    regions=smoke_regions,
                    smoke_mask=smoke_mask,
                    frame_bgr=frame,
                    max_aspect_ratio=smoke_max_aspect_ratio,
                    max_edge_density=smoke_max_edge_density,
                )

            # TV4 Temporal: theo dõi và bỏ phiếu theo thời gian.

            confirmed_fire = fire_temporal.update(fire_regions, frame_no)

            confirmed_smoke = smoke_temporal.update(smoke_regions, frame_no)

            if confirmed_fire:
                last_fire_region = union_regions(confirmed_fire, "fire")[0]
                last_fire_frame = frame_no
            elif (
                single_box_per_kind
                and last_fire_region is not None
                and frame_no - last_fire_frame == 1
            ):
                x, y, w, h = last_fire_region.bbox
                fire_color_roi = colors["fire"][y : y + h, x : x + w]
                if fire_color_roi.size and cv2.countNonZero(fire_color_roi) >= max(
                    20, int(0.02 * fire_color_roi.size)
                ):
                    confirmed_fire = [last_fire_region]

            confirmed = confirmed_fire + confirmed_smoke

            if single_box_per_kind:
                confirmed = union_regions(confirmed_fire, "fire") + union_regions(
                    confirmed_smoke, "smoke"
                )

            # Vẽ vùng đã xác nhận lên frame kết quả.

            annotated = frame.copy()

            timestamp = frame_no / fps

            for obj in confirmed:
                x, y, w, h = obj.bbox

                color = (0, 0, 255) if obj.kind == "fire" else (255, 200, 0)

                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

                cv2.putText(
                    annotated,
                    obj.kind.upper(),
                    (x, max(y - 7, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                events.append(
                    [
                        frame_no,
                        round(timestamp, 3),
                        obj.kind,
                        x,
                        y,
                        w,
                        h,
                        obj.score,
                    ]
                )

            writer.write(annotated)

            if dashboard_writer is not None:
                dashboard_writer.write(
                    compose_pipeline_frame(
                        {
                            "fire_color": colors["fire"],
                            "smoke_color": colors["smoke"],
                            "fire_motion": fire_movement,
                            "smoke_motion": smoke_movement,
                            "fire_fusion": fire_mask,
                            "smoke_fusion": smoke_mask,
                            "fire_regions": draw_regions(
                                frame, fire_regions, show_contour_info=False
                            ),
                            "smoke_regions": draw_regions(
                                frame, smoke_regions, show_contour_info=False
                            ),
                            "fire_temporal": draw_regions(
                                frame, confirmed_fire, show_contour_info=False
                            ),
                            "smoke_temporal": draw_regions(
                                frame, confirmed_smoke, show_contour_info=False
                            ),
                        },
                        frame_no,
                        fps,
                    )
                )

            frame_no += 1

    finally:
        cap.release()
        writer.release()
        if dashboard_writer is not None:
            dashboard_writer.release()

    if frame_no == 0:
        raise ValueError(f"Video contains no decodable frames: {input_path}")

    # Ghi từng vùng theo frame vào CSV.

    if csv_path:
        csv_file = Path(csv_path)

        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with csv_file.open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(
                [
                    "frame",
                    "timestamp_sec",
                    "kind",
                    "x",
                    "y",
                    "w",
                    "h",
                    "score",
                ]
            )

            csv_writer.writerows(events)

    result = {
        "frames": frame_no,
        "event_rows": len(events),
        "output_video": str(out_path),
    }
    if dashboard_path is not None:
        result["dashboard_video"] = str(dashboard_path)
    return result


# Giao diện dòng lệnh.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FireGuard FIRE/SMOKE detection")

    parser.add_argument("--input", required=True)

    parser.add_argument("--output", default="outputs/videos/out.mp4")

    parser.add_argument("--csv", default="outputs/events/events.csv")

    parser.add_argument(
        "--fire-fusion",
        choices=FUSION_MODES,
        default=DEMO_OPTIONS["fire_fusion"],
    )

    parser.add_argument(
        "--smoke-fusion",
        choices=FUSION_MODES,
        default=None,
    )

    parser.add_argument(
        "--motion-dilate",
        type=int,
        default=DEMO_OPTIONS["motion_dilate"],
    )

    parser.add_argument(
        "--smoke-motion",
        choices=("frame_diff", "mog2"),
        default=DEMO_OPTIONS["smoke_motion"],
    )

    parser.add_argument(
        "--smoke-motion-dilate",
        type=int,
        default=DEMO_OPTIONS["smoke_motion_dilate"],
    )

    # Các tham số ghép vùng khói.

    parser.add_argument("--smoke-region-close", type=int, default=15)

    parser.add_argument("--smoke-region-motion-ratio", type=float, default=0.015)

    parser.add_argument("--smoke-region-min-motion-pixels", type=int, default=12)

    parser.add_argument("--smoke-region-min-area", type=int, default=130)
    parser.add_argument("--smoke-region-max-area-ratio", type=float, default=0.50)
    parser.add_argument("--smoke-growth-kernel", type=int, default=5)
    parser.add_argument("--smoke-growth-iterations", type=int, default=8)

    # Quyết định theo thời gian cho khói.

    parser.add_argument(
        "--smoke-temporal-mode",
        choices=("vote", "hybrid"),
        default=DEMO_OPTIONS["smoke_temporal_mode"],
    )

    # Bộ lọc hình dạng/cạnh khói.

    parser.add_argument(
        "--smoke-verify",
        choices=SMOKE_VERIFY_MODES,
        default=DEMO_OPTIONS["smoke_verify"],
        help="Optional SMOKE candidate verification",
    )

    parser.add_argument(
        "--smoke-max-aspect-ratio",
        type=float,
        default=DEMO_OPTIONS["smoke_max_aspect_ratio"],
    )

    parser.add_argument(
        "--smoke-max-edge-density",
        type=float,
        default=DEMO_OPTIONS["smoke_max_edge_density"],
    )

    parser.add_argument(
        "--single-box-per-kind",
        action="store_true",
        default=DEMO_OPTIONS["single_box_per_kind"],
        help="Draw one enclosing box per kind for confirmed regions",
    )
    parser.add_argument(
        "--multiple-boxes",
        action="store_false",
        dest="single_box_per_kind",
        help="Keep separate confirmed boxes for diagnostics",
    )

    parser.add_argument(
        "--smoke-engine", choices=("classical", "opencv_context"), default="classical"
    )
    parser.add_argument("--smoke-vote-hits", type=int)

    args = parser.parse_args()

    result = run_video(
        args.input,
        args.output,
        args.csv,
        fire_fusion=args.fire_fusion,
        smoke_fusion=(
            args.smoke_fusion
            or (
                "adaptive_region_motion"
                if args.smoke_engine == "opencv_context"
                else DEMO_OPTIONS["smoke_fusion"]
            )
        ),
        motion_dilate=args.motion_dilate,
        smoke_motion=args.smoke_motion,
        smoke_motion_dilate=args.smoke_motion_dilate,
        smoke_region_close=args.smoke_region_close,
        smoke_region_motion_ratio=args.smoke_region_motion_ratio,
        smoke_region_min_motion_pixels=(args.smoke_region_min_motion_pixels),
        smoke_region_min_area=args.smoke_region_min_area,
        smoke_region_max_area_ratio=args.smoke_region_max_area_ratio,
        smoke_growth_kernel=args.smoke_growth_kernel,
        smoke_growth_iterations=args.smoke_growth_iterations,
        smoke_temporal_mode=args.smoke_temporal_mode,
        smoke_verify=args.smoke_verify,
        smoke_max_aspect_ratio=args.smoke_max_aspect_ratio,
        smoke_max_edge_density=args.smoke_max_edge_density,
        single_box_per_kind=args.single_box_per_kind,
        smoke_engine=args.smoke_engine,
        smoke_vote_hits=args.smoke_vote_hits,
    )

    print(result)
