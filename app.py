"""Giao diện Streamlit cho pipeline phát hiện lửa và khói bằng OpenCV."""

import csv
import hashlib
import io
import tempfile
from pathlib import Path
import subprocess
import shutil

import streamlit as st
import streamlit.components.v1 as components

from fireguard.pipeline import run_video
from fireguard.presets import DEMO_OPTIONS, OPENCV_CONTEXT_OPTIONS
from fireguard.video_sync import synchronized_video_html

st.set_page_config(
    page_title="FireGuard | Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
)

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

st.title("🔥 FireGuard")

st.caption("Phát hiện vùng nghi lửa và khói trong video bằng OpenCV.")

st.divider()

PIPELINE_OPTIONS = {**DEMO_OPTIONS, **OPENCV_CONTEXT_OPTIONS}
DISPLAY_LAYOUT_VERSION = 4


def find_usable_ffmpeg():
    """Tìm FFmpeg chạy được để xuất MP4 tương thích trình duyệt."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    try:
        probe = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return executable if probe.returncode == 0 else None


def convert_for_browser(ffmpeg_exe: str, source: Path, destination: Path) -> bytes:
    """Encode an OpenCV video for browser playback."""
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(source),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination.read_bytes()


if "fireguard_result" not in st.session_state:
    st.session_state.fireguard_result = None

if "fireguard_input_id" not in st.session_state:
    st.session_state.fireguard_input_id = None


def analyze_csv(csv_bytes: bytes, total_frames: int):
    """Đếm số hộp và số frame có phát hiện theo từng loại."""
    text = csv_bytes.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))

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


st.subheader("1. Tải video đầu vào")

uploaded = st.file_uploader(
    "Chọn video cần phân tích",
    type=["mp4", "avi", "mov"],
    help="Hỗ trợ video MP4, AVI và MOV.",
)

if uploaded is not None:

    video_bytes = uploaded.getvalue()

    input_id = hashlib.sha256(video_bytes).hexdigest()

    st.video(video_bytes)

    st.subheader("2. Phân tích video")

    if (
        st.session_state.fireguard_input_id != input_id
        or (
            st.session_state.fireguard_result is not None
            and st.session_state.fireguard_result.get("layout_version") != DISPLAY_LAYOUT_VERSION
        )
    ):
        st.session_state.fireguard_result = None
        st.session_state.fireguard_input_id = input_id

    with st.expander("Xem cấu hình thuật toán", expanded=False):

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**FIRE Detection**")

            st.write("Color: HSV + Fire Core")

            st.write("Motion: Frame Difference")

            st.write("Fusion: AND + Dilation 5×5")

            st.write("Temporal: Voting")

        with col_b:
            st.markdown("**SMOKE Detection**")
            st.write("TV1: màu/độ mịn theo bối cảnh trời hoặc nền")
            st.write("TV2: MOG2; Fusion: vùng chuyển động hoặc màu")
            st.write("TV3: morphology + kiểm tra vùng")
            st.write("TV4: bỏ phiếu 5/7 frame")

    ffmpeg_exe = find_usable_ffmpeg()
    if ffmpeg_exe is None:
        st.warning("Cần cài FFmpeg và thêm vào PATH để xuất video kết quả H.264.")

    if st.button(
        "▶ Bắt đầu phân tích video",
        type="primary",
        use_container_width=True,
        disabled=ffmpeg_exe is None,
    ):

        # Xóa kết quả cũ trước khi xử lý video mới.
        st.session_state.fireguard_result = None

        try:

            with st.spinner("Đang xử lý video bằng FireGuard..."):

                # Đọc hết kết quả trước khi thư mục tạm được giải phóng.
                with tempfile.TemporaryDirectory() as temp_dir:

                    workdir = Path(temp_dir)

                    suffix = Path(uploaded.name).suffix.lower() or ".mp4"

                    input_path = workdir / f"input{suffix}"

                    output_path = workdir / "fireguard_result.mp4"

                    dashboard_path = workdir / "fireguard_dashboard.mp4"

                    csv_path = workdir / "fireguard_events.csv"

                    input_path.write_bytes(video_bytes)

                    result = run_video(
                        input_path=str(input_path),
                        output_path=str(output_path),
                        csv_path=str(csv_path),
                        dashboard_path=str(dashboard_path),
                        **PIPELINE_OPTIONS,
                    )

                    # Hai video có cùng FPS và số frame; video kết quả điều khiển video các bước.
                    dashboard_video = convert_for_browser(
                        ffmpeg_exe, dashboard_path, workdir / "dashboard_browser.mp4"
                    )
                    processed_video = convert_for_browser(
                        ffmpeg_exe, output_path, workdir / "fireguard_browser.mp4"
                    )

                    processed_csv = csv_path.read_bytes()

                    stats = analyze_csv(processed_csv, result["frames"])

                    st.session_state.fireguard_result = {
                        "input_id": input_id,
                        "layout_version": DISPLAY_LAYOUT_VERSION,
                        "video": processed_video,
                        "dashboard_video": dashboard_video,
                        "csv": processed_csv,
                        "stats": stats,
                        "filename": f"{Path(uploaded.name).stem}_opencv_smoke",
                    }

            st.success("Phân tích video hoàn tất!")

        except Exception as exc:

            st.session_state.fireguard_result = None

            st.error(f"Không thể xử lý video: {exc}")

    saved_result = st.session_state.fireguard_result

    if saved_result is not None and saved_result["input_id"] == input_id:

        st.divider()

        st.subheader("3. Kết quả phát hiện")

        stats = saved_result["stats"]

        metric_a, metric_b, metric_c = st.columns(3)

        with metric_a:
            st.metric("Tổng số frame", stats["total_frames"])

        with metric_b:
            st.metric("Frame phát hiện FIRE", stats["fire_frames"])

        with metric_c:
            st.metric("Frame phát hiện SMOKE", stats["smoke_frames"])

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

        st.markdown("#### Video sau xử lý và các bước")

        components.html(
            synchronized_video_html(
                saved_result["video"], saved_result["dashboard_video"]
            ),
            height=700,
            scrolling=True,
        )

        st.caption(
            "Bên trái là video kết quả; bên phải là các bước FIRE/SMOKE. "
            "Phát, dừng hoặc tua ở video kết quả để đồng bộ video các bước. "
            "TV1/TV2/Fusion là mask; TV3 và Temporal hiển thị vùng."
        )

        st.divider()

        st.subheader("4. Tải kết quả")

        download_a, download_b, download_c = st.columns(3)

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

        with download_c:

            st.download_button(
                label="⬇ Tải video các bước (.mp4)",
                data=saved_result["dashboard_video"],
                file_name=f"{filename}_pipeline.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

else:

    st.info("Tải một video để bắt đầu sử dụng FireGuard.")
