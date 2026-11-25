"""Two browser video players controlled by the processed result timeline."""

import base64


def synchronized_video_html(result_video: bytes, steps_video: bytes) -> str:
    """Build two separate players; only the result player has controls."""
    result_b64 = base64.b64encode(result_video).decode("ascii")
    steps_b64 = base64.b64encode(steps_video).decode("ascii")
    page = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: sans-serif; background: transparent; }
  .players { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  .panel { min-width: 0; padding: 12px; border-radius: 12px; background: #15191e; color: white; }
  .panel h3 { margin: 0 0 10px; font-size: 16px; font-weight: 600; }
  video { width: 100%; display: block; background: #000; border-radius: 7px; }
  #result { aspect-ratio: 16 / 9; }
  #steps { aspect-ratio: 720 / 648; pointer-events: none; }
  #message { color: #ffca7a; margin: 9px 0 0; font-size: 13px; }
  @media (max-width: 700px) { .players { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="players">
  <section class="panel">
    <h3>Video sau xử lý · điều khiển tại đây</h3>
    <video id="result" controls playsinline preload="auto"></video>
  </section>
  <section class="panel">
    <h3>Các bước FIRE / SMOKE · theo video kết quả</h3>
    <video id="steps" muted playsinline preload="auto" aria-label="Các bước xử lý"></video>
  </section>
</div>
<p id="message" role="status"></p>
<script>
  const result = document.getElementById("result");
  const steps = document.getElementById("steps");
  const message = document.getElementById("message");

  // Blob URLs allow seeking even when the MP4 files are passed through Streamlit.
  function videoUrl(encoded) {
    const chunks = [];
    for (let offset = 0; offset < encoded.length; offset += 32768) {
      const binary = atob(encoded.slice(offset, offset + 32768));
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      chunks.push(bytes);
    }
    return URL.createObjectURL(new Blob(chunks, { type: "video/mp4" }));
  }
  const resultUrl = videoUrl("__RESULT_VIDEO__");
  const stepsUrl = videoUrl("__STEPS_VIDEO__");
  result.src = resultUrl;
  steps.src = stepsUrl;

  function align(force = false) {
    if (steps.readyState < 1 || !Number.isFinite(result.currentTime)) return;
    if (force || Math.abs(steps.currentTime - result.currentTime) > 0.15) {
      steps.currentTime = result.currentTime;
    }
  }
  function playSteps() {
    if (result.paused || result.ended || steps.readyState < 1) return;
    align(true);
    steps.playbackRate = result.playbackRate;
    const playback = steps.play();
    if (playback) playback.then(
      () => { message.textContent = ""; },
      (error) => {
        if (error.name !== "AbortError") {
          message.textContent = "Không thể phát video các bước trong trình duyệt này.";
        }
      }
    );
  }
  result.addEventListener("play", playSteps);
  result.addEventListener("playing", () => { if (steps.paused) playSteps(); });
  result.addEventListener("pause", () => { steps.pause(); align(true); });
  result.addEventListener("waiting", () => steps.pause());
  result.addEventListener("seeking", () => { steps.pause(); align(true); });
  result.addEventListener("seeked", () => { align(true); playSteps(); });
  result.addEventListener("timeupdate", () => align());
  result.addEventListener("ratechange", () => { steps.playbackRate = result.playbackRate; });
  result.addEventListener("ended", () => { steps.pause(); align(true); });
  steps.addEventListener("loadedmetadata", () => { align(true); playSteps(); });
  window.addEventListener("beforeunload", () => {
    URL.revokeObjectURL(resultUrl);
    URL.revokeObjectURL(stepsUrl);
  });
</script>
</body>
</html>"""
    return page.replace("__RESULT_VIDEO__", result_b64).replace("__STEPS_VIDEO__", steps_b64)
