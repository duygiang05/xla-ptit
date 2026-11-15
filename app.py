
"""FireGuard - Streamlit demonstration application.

Uses the integrated OpenCV pipeline without modifying
the detection algorithms.

FIRE:
    Frame Difference + AND Dilation 5 + Temporal Vote

SMOKE:
    MOG2 + Region Motion Fusion
    + Shape/Edge Verification + Temporal Vote
"""

import csv
import hashlib
import io
import tempfile
from pathlib import Path
import subprocess
import shutil

import streamlit as st

from fireguard.pipeline import run_video


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="FireGuard | Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.4px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 14px;
    }

    div.stButton > button[kind="primary"] {
        border-radius: 9px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Application header
# ============================================================

st.title("🔥 FireGuard")

st.markdown(
    "**Hệ thống phát hiện vùng nghi lửa và khói trong video "
    "bằng các kỹ thuật xử lý ảnh OpenCV.**"
)

st.caption(
    "Bài tập lớn Xử lý ảnh | Color Detection • Motion Detection "
    "• Fusion • Morphology • Temporal Filtering"
)

st.divider()


# ============================================================
# Pipeline configuration
# ============================================================

PIPELINE_OPTIONS = {
    "fire_fusion": "and_dilate",
    "smoke_fusion": "region_motion",
    "motion_dilate": 5,
    "smoke_motion": "mog2",
    "smoke_motion_dilate": 9,
    "smoke_temporal_mode": "vote",
    "smoke_verify": "shape_edge",
    "smoke_max_aspect_ratio": 1.10,
    "smoke_max_edge_density": 0.02,
}


# ============================================================
# Session state
# ============================================================

if "fireguard_result" not in st.session_state:
    st.session_state.fireguard_result = None

if "fireguard_input_id" not in st.session_state:
    st.session_state.fireguard_input_id = None


# ============================================================
# Helper: CSV statistics
# ============================================================

def analyze_csv(csv_bytes: bytes, total_frames: int):
    text = csv_bytes.decode("utf-8-sig")

    reader = csv.DictReader(
        io.StringIO(text)
    )

    fire_boxes = 0
    smoke_boxes = 0

    fire_frames = set()
    smoke_frames = set()

    for row in reader:
        kind = row["kind"].lower()
        frame_number = int(row["frame"])

        if kind == "fire":
            fire_boxes += 1
            fire_frames.add(frame_number)

        elif kind == "smoke":
            smoke_boxes += 1
            smoke_frames.add(frame_number)

    return {
        "total_frames": total_frames,
        "fire_boxes": fire_boxes,
        "smoke_boxes": smoke_boxes,
        "fire_frames": len(fire_frames),
        "smoke_frames": len(smoke_frames),
    }


# ============================================================
# Video upload
# ============================================================

st.subheader("1. Tải video đầu vào")

uploaded = st.file_uploader(
    "Chọn video cần phân tích",
    type=["mp4", "avi", "mov"],
    help="Hỗ trợ video MP4, AVI và MOV.",
)

if uploaded is not None:

    video_bytes = uploaded.getvalue()

    input_id = hashlib.sha256(
        video_bytes
    ).hexdigest()

    # Remove old results if a different video is uploaded.
    if st.session_state.fireguard_input_id != input_id:
        st.session_state.fireguard_result = None
        st.session_state.fireguard_input_id = input_id

    st.success(
        f"Đã tải lên: {uploaded.name}"
    )

    st.video(video_bytes)

    # ========================================================
    # Analysis settings
    # ========================================================

    st.subheader("2. Cấu hình xử lý")

    with st.expander(
        "Xem cấu hình thuật toán",
        expanded=False,
    ):

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**FIRE Detection**")

            st.write(
                "Color: HSV + Fire Core"
            )

            st.write(
                "Motion: Frame Difference"
            )

            st.write(
                "Fusion: AND + Dilation 5×5"
            )

            st.write(
                "Temporal: Voting"
            )

        with col_b:
            st.markdown("**SMOKE Detection**")

            st.write(
                "Color: HSV + Texture"
            )

            st.write(
                "Motion: MOG2"
            )

            st.write(
                "Fusion: Region-Level Motion"
            )

            st.write(
                "Verification: Shape + Edge"
            )

            st.write(
                "Temporal: Voting"
            )

    st.info(
        "FireGuard xử lý cả FIRE và SMOKE trong cùng một video. "
        "Các vùng phát hiện là kết quả thuật toán, "
        "không phải xác suất dự đoán đã hiệu chuẩn."
    )

    # ========================================================
    # Run analysis
    # ========================================================

    if st.button(
        "▶ Bắt đầu phân tích video",
        type="primary",
        use_container_width=True,
    ):

        # Clear old output before starting new analysis.
        st.session_state.fireguard_result = None

        try:

            with st.spinner(
                "Đang xử lý video bằng FireGuard..."
            ):

                # All temporary files remain available until
                # their contents are read into session state.
                with tempfile.TemporaryDirectory() as temp_dir:

                    workdir = Path(temp_dir)

                    suffix = (
                        Path(uploaded.name).suffix.lower()
                        or ".mp4"
                    )

                    input_path = (
                        workdir / f"input{suffix}"
                    )

                    output_path = (
                        workdir / "fireguard_result.mp4"
                    )

                    csv_path = (
                        workdir / "fireguard_events.csv"
                    )

                    input_path.write_bytes(
                        video_bytes
                    )

                    # Use the validated integration settings.
                    result = run_video(
                        input_path=str(input_path),
                        output_path=str(output_path),
                        csv_path=str(csv_path),
                        **PIPELINE_OPTIONS,
                    )

                    # Read outputs before temporary cleanup.
                    # Convert OpenCV MP4V output to browser-compatible H.264
                    browser_video_path = workdir / "fireguard_browser.mp4"

                    ffmpeg_exe = shutil.which("ffmpeg")

                    if ffmpeg_exe is None:
                        raise RuntimeError("Không tìm thấy FFmpeg trong PATH")

                    subprocess.run(
                        [
                            ffmpeg_exe,
                            "-y",
                            "-loglevel", "error",
                            "-i", str(output_path),
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "-an",
                            str(browser_video_path),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    processed_video = browser_video_path.read_bytes()

                    processed_csv = (
                        csv_path.read_bytes()
                    )

                    stats = analyze_csv(
                        processed_csv,
                        result["frames"],
                    )

                    st.session_state.fireguard_result = {
                        "input_id": input_id,
                        "video": processed_video,
                        "csv": processed_csv,
                        "stats": stats,
                        "filename": Path(uploaded.name).stem,
                    }

            st.success(
                "Phân tích video hoàn tất!"
            )

        except Exception as exc:

            st.session_state.fireguard_result = None

            st.error(
                f"Không thể xử lý video: {exc}"
            )

    # ========================================================
    # Display results
    # ========================================================

    saved_result = st.session_state.fireguard_result

    if (
        saved_result is not None
        and saved_result["input_id"] == input_id
    ):

        st.divider()

        st.subheader("3. Kết quả phát hiện")

        stats = saved_result["stats"]

        metric_a, metric_b, metric_c = st.columns(3)

        with metric_a:
            st.metric(
                "Tổng số frame",
                stats["total_frames"],
            )

        with metric_b:
            st.metric(
                "Frame phát hiện FIRE",
                stats["fire_frames"],
            )

        with metric_c:
            st.metric(
                "Frame phát hiện SMOKE",
                stats["smoke_frames"],
            )

        st.markdown("#### Thống kê bounding box")

        col_fire, col_smoke = st.columns(2)

        with col_fire:
            st.metric(
                "🔥 FIRE",
                stats["fire_boxes"],
                help="Tổng lượt bounding box FIRE trên các frame.",
            )

        with col_smoke:
            st.metric(
                "💨 SMOKE",
                stats["smoke_boxes"],
                help="Tổng lượt bounding box SMOKE trên các frame.",
            )

        st.markdown("#### Video sau xử lý")

        st.video(
            saved_result["video"],
            format="video/mp4",
        )

        st.caption(
            "Bounding box màu đỏ: FIRE. "
            "Bounding box màu xanh: SMOKE."
        )

        st.divider()

        st.subheader("4. Tải kết quả")

        download_a, download_b = st.columns(2)

        filename = saved_result["filename"]

        with download_a:

            st.download_button(
                label="⬇ Tải video kết quả (.mp4)",
                data=saved_result["video"],
                file_name=f"{filename}_fireguard.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        with download_b:

            st.download_button(
                label="⬇ Tải dữ liệu phát hiện (.csv)",
                data=saved_result["csv"],
                file_name=f"{filename}_fireguard.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.caption(
            "CSV chứa frame, thời điểm, loại đối tượng, "
            "tọa độ bounding box và score của Region."
        )

else:

    st.info(
        "Tải một video để bắt đầu sử dụng FireGuard."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "FireGuard | Bài tập lớn Xử lý ảnh | "
    "OpenCV + Python + Streamlit"
)
