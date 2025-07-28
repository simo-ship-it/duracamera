"""
durastream.py – People counter con streaming MJPEG
Dipendenze: ultralytics, opencv-python-headless, yt-dlp, flask
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
VIDEO_URL = os.getenv("VIDEO_URL",
                      "https://www.youtube.com/watch?v=Z49UkOi08DE")


def cap_from_youtube(url: str,
                     quality: str = "best[height<=720]") -> cv2.VideoCapture:
    """Restituisce uno stream OpenCV dal video YouTube."""
    with YoutubeDL({"format": quality, "quiet": True}) as ydl:
        stream_url = ydl.extract_info(url, download=False)["url"]

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire lo stream YouTube")
    return cap


# ------- inizializzazione -------
model = YOLO(MODEL_PATH)
cap = cap_from_youtube(VIDEO_URL)
prev_t = time.perf_counter()

# ------- Flask -------
app = Flask(__name__)


def frame_generator():
    """Produce i frame JPEG in multipart/x-mixed-replace."""
    global prev_t
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO inference
        results = model.predict(frame, conf=CONF_THRES, device="cpu",
                                verbose=False)
        dets = results[0].boxes

        # Disegna riquadri
        people = 0
        for box in dets:
            if model.names[int(box.cls[0])] == TARGET:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS & overlay
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Codifica JPEG & streaming
        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    """Pagina HTML minimale con lo stream video."""
    return render_template_string("""
<!doctype html>
<title>People Counter</title>
<style>body{margin:0;background:#000;display:flex;justify-content:center;
align-items:center;height:100vh}</style>
<img src="{{ url_for('video_feed') }}" style="max-width:100%;height:auto">
""")


@app.route("/video")
def video_feed():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            threaded=True)