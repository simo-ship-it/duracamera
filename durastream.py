from __future__ import annotations
import os, sys, time, json, atexit, collections
import cv2
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO
from yt_dlp import YoutubeDL

# --------- Env config ---------
TARGET       = "person"
YOLO_MODEL   = os.getenv("YOLO_MODEL", "yolov10s.pt")
YOLO_DEVICE  = os.getenv("YOLO_DEVICE", "cpu")
CONF_THRES   = float(os.getenv("CONF", 0.4))
VIDEO_URL    = os.getenv("VIDEO_URL", "https://www.youtube.com/watch?v=Z49UkOi08DE")
YT_FORMAT    = os.getenv("YT_FORMAT", "best[ext=mp4][height<=720]/best")

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")  # sopprime warning Ultralytics

# --------- Helper functions ---------

def _pick_stream_url(info: dict) -> str | None:
    """Seleziona un URL video HTTPS MP4 o DASH leggibile da OpenCV."""
    if info.get("url") and info.get("ext") == "mp4":
        return info["url"]
    fmts = [
        f for f in info.get("formats", [])
        if f.get("vcodec") != "none" and f.get("protocol", "").startswith("https")
    ]
    fmts.sort(key=lambda f: f.get("height", 0), reverse=True)
    return fmts[0]["url"] if fmts else None


def cap_from_youtube(url: str) -> cv2.VideoCapture:
    """Restituisce un VideoCapture aprendo uno stream YouTube via yt‑dlp."""
    with YoutubeDL({"format": YT_FORMAT, "quiet": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = _pick_stream_url(info)
        if not stream_url:
            sys.stderr.write("[durastream] Nessun formato video compatibile. Dump<=5 formats:\n" +
                             json.dumps(info.get("formats", [])[:5], indent=2) + "\n")
            raise RuntimeError("Nessun formato video compatibile per la URL fornita")
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire lo stream video YouTube")
    return cap


def cap_from_source(url: str) -> cv2.VideoCapture:
    """Restituisce un VideoCapture da URL YouTube o stream diretto (MJPEG/RTSP)."""
    if "youtu" in url:
        return cap_from_youtube(url)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire lo stream: {url}")
    return cap

# --------- YOLO init ---------
model = YOLO(YOLO_MODEL, task="detect")
cap   = cap_from_source(VIDEO_URL)
atexit.register(lambda: cap.release())  # rilascia la camera alla chiusura

prev_t = time.perf_counter()
fps_window = collections.deque(maxlen=30)  # media mobile FPS
last_people = 0
last_fps = 0.0
last_ts = time.time()

# --------- Flask app ---------
app = Flask(__name__)


def frame_generator():
    global prev_t, last_people, last_fps, last_ts, cap
    while True:
        ok, frame = cap.read()
        if not ok:
            # Riconnessione automatica se lo stream cade
            time.sleep(2)
            try:
                new_cap = cap_from_source(VIDEO_URL)
                cap.release()
                cap = new_cap
                continue
            except Exception as e:
                sys.stderr.write(f"[durastream] reconnect failed: {e}\n")
                continue

        results = model.predict(frame, conf=CONF_THRES, device=YOLO_DEVICE, verbose=False)
        dets = results[0].boxes

        people = 0
        for box in dets:
            if model.names[int(box.cls[0])] == TARGET:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.perf_counter()
        inst_fps = 1.0 / (now - prev_t)
        prev_t = now
        fps_window.append(inst_fps)
        fps = sum(fps_window) / len(fps_window)

        last_people, last_fps, last_ts = people, fps, time.time()

        cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template_string(
        """
<!doctype html>
<title>duracamera v10</title>
<style>
  body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}
</style>
<img src="{{ url_for('video_feed') }}" style="max-width:100%;height:auto">
"""
    )


@app.route("/video")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/count")
def api_count():
    return jsonify(people=last_people, fps=round(last_fps, 2), ts=int(last_ts))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), threaded=True)