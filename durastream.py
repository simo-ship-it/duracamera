"""
durastream.py – streaming YouTube → MJPEG con YOLOv8
Fix 2 (2025‑07‑29):
  • gestisce video con formati DASH/segmented e sceglie il primo MP4 progressivo.
  • logga i formati se non ne trova di compatibili.
  • sopprime warning Ultralytics impostando YOLO_CONFIG_DIR=/tmp (già nel Dockerfile).
"""

import os
import sys
import time
import json
import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from yt_dlp import YoutubeDL

TARGET = "person"
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.4
VIDEO_URL = os.getenv("VIDEO_URL", "https://www.youtube.com/watch?v=Z49UkOi08DE")

# Inibisci i warning Ultralytics (cartella non scrivibile)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")


def pick_stream_url(info: dict) -> str | None:
    """Ritorna un URL riproducibile da OpenCV (HTTPS, audio+video MP4)."""
    # 1️⃣ Alcune versioni yt-dlp restituiscono url diretto
    if "url" in info and info.get("ext") == "mp4":
        return info["url"]

    # 2️⃣ Scorri i formati e scegli MP4 progressivo ≤720p (acodec+vcodec non 'none')
    fmts = info.get("formats", [])
    progressive = [
        f for f in fmts
        if f.get("ext") == "mp4"
        and f.get("acodec") != "none"
        and f.get("vcodec") != "none"
        and f.get("protocol", "").startswith("https")
    ]
    if progressive:
        # pick the one with the highest resolution up to 720p
        progressive.sort(key=lambda f: f.get("height", 0), reverse=True)
        return progressive[0]["url"]

    # 3️⃣ Fallback: qualsiasi formato con chiave 'url'
    for f in fmts:
        if f.get("url"):
            return f["url"]
    return None


def cap_from_youtube(url: str) -> cv2.VideoCapture:
    """Ritorna un VideoCapture o solleva RuntimeError se fallisce."""
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "quiet": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = pick_stream_url(info)
        if not stream_url:
            # Log formati disponibili per debug
            sys.stderr.write("[durastream] Nessun formato MP4 progressivo trovato. Formati disponibili:\n")
            sys.stderr.write(json.dumps(info.get("formats", [])[:10], indent=2) + "\n")
            raise RuntimeError("Nessun formato video compatibile trovato per la URL fornita")

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire lo stream video")
    return cap


# --------------------- YOLO & Flask ---------------------
model = YOLO(MODEL_PATH)
cap = cap_from_youtube(VIDEO_URL)
prev_t = time.perf_counter()

app = Flask(__name__)


def frame_generator():
    global prev_t
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=CONF_THRES, device="cpu", verbose=False)
        dets = results[0].boxes

        people = 0
        for box in dets:
            if model.names[int(box.cls[0])] == TARGET:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


@app.route("/")

def index():
    return render_template_string("""
<!doctype html>
<title>duracamera</title>
<style>body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}</style>
<img src="{{ url_for('video_feed') }}" style="max-width:100%;height:auto">
""")


@app.route("/video")

def video_feed():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), threaded=True)
