"""
durastream.py – streaming YouTube → MJPEG con YOLOv8
Fix 3 (2025‑07‑29):
  • Supporta stream *video‑only* (acodec=none) così OpenCV legge anche formati DASH.
  • Logga comunque i primi 5 formati in caso di fallimento.
  • Variabile d’ambiente facoltativa YT_FORMAT per override manuale.
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
YT_FORMAT = os.getenv("YT_FORMAT", "best[ext=mp4][height<=720]/best")

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")  # sopprime warning Ultralytics


def pick_stream_url(info: dict) -> str | None:
    """Ritorna un URL HTTPS MP4 leggibile da OpenCV.
    Non serve audio: accetta acodec=='none'.
    """
    if "url" in info and info.get("ext") == "mp4":
        return info["url"]

    fmts = info.get("formats", [])
    mp4_fmts = [
        f for f in fmts
        if f.get("ext") == "mp4" and f.get("vcodec") != "none" and f.get("protocol", "").startswith("https")
    ]
    if mp4_fmts:
        mp4_fmts.sort(key=lambda f: f.get("height", 0), reverse=True)
        return mp4_fmts[0]["url"]

    # fallback qualsiasi formato HTTPS video-only
    for f in fmts:
        if f.get("url") and f.get("vcodec") != "none":
            return f["url"]
    return None


def cap_from_youtube(url: str) -> cv2.VideoCapture:
    ydl_opts = {"format": YT_FORMAT, "quiet": True, "noplaylist": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = pick_stream_url(info)
        if not stream_url:
            sys.stderr.write("[durastream] Nessun formato MP4/DASH compatibile. Prime 5 entry formats:\n")
            sys.stderr.write(json.dumps(info.get("formats", [])[:5], indent=2) + "\n")
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
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


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
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), threaded=True)
#programma funzionante numero 1