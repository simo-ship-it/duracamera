"""
durastream.py – streaming YouTube → MJPEG con YOLOv8
Fix: gestione KeyError 'url' (yt‑dlp 2024+) e noplaylist.
"""

import os
import time
import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
from yt_dlp import YoutubeDL

TARGET = "person"
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.4
VIDEO_URL = os.getenv("VIDEO_URL", "https://www.youtube.com/watch?v=Z49UkOi08DE")


def cap_from_youtube(url: str, quality: str = "best[height<=720]") -> cv2.VideoCapture:
    """Restituisce un VideoCapture a partire da un link YouTube.

    yt‑dlp >=2024 non sempre include la chiave 'url' top‑level;
    la ricaviamo dal formato scelto.
    """
    ydl_opts = {
        "format": quality,
        "quiet": True,
        "noplaylist": True,  # evita di trattare l'URL come playlist
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # Caso 1: extractor restituisce url diretto (alcune versioni / scenari)
        if "url" in info:
            stream_url = info["url"]
        else:
            # Caso 2: scegliere il primo formato con protocollo HTTPS
            fmts = info.get("formats")
            if not fmts:
                raise RuntimeError("Nessun formato video disponibile per la URL fornita")
            stream_url = next((f["url"] for f in fmts if f.get("url")), fmts[0]["url"])

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
