"""Minimal Streamlit demo: session-safe download of processed outputs."""
import tempfile
from pathlib import Path
import streamlit as st
from fireguard.pipeline import run_video

st.set_page_config(page_title="FireGuard – BTL XLA", layout="centered")
st.title("FireGuard – Phát hiện vùng nghi ngờ khói/lửa")
st.caption("Prototype học thuật. Nhãn phát hiện chỉ là giả thuyết, chưa đánh giá độ chính xác.")
uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov"])
if uploaded is not None:
    video_bytes = uploaded.getvalue()
    st.video(video_bytes)
    if st.button("Phân tích video"):
        try:
            with st.spinner("Đang xử lý..."):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    inp = root / ("input" + Path(uploaded.name).suffix)
                    out = root / "result.mp4"
                    csv_file = root / "events.csv"
                    inp.write_bytes(video_bytes)
                    result = run_video(str(inp), str(out), str(csv_file))
                    st.session_state["result_video"] = out.read_bytes()
                    st.session_state["result_csv"] = csv_file.read_bytes()
                    st.session_state["result_summary"] = result
        except Exception as exc:
            st.error(f"Không xử lý được video: {exc}")
    if "result_video" in st.session_state:
        st.write(st.session_state["result_summary"])
        st.download_button("Tải video kết quả", st.session_state["result_video"], file_name="fireguard_result.mp4", mime="video/mp4")
        st.download_button("Tải CSV", st.session_state["result_csv"], file_name="events.csv", mime="text/csv")
        st.caption("Một số trình duyệt không xem trực tiếp được MP4 codec mp4v của OpenCV; có thể tải file về để xem.")
