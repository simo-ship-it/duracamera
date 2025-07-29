"""
durastream.py – YOLOv10 + MJPEG API (da IP camera)
--------------------------------------------------
• /video        → stream MJPEG con riquadri persone
• /api/count    → JSON {people, fps, ts}

▶ CONFIGURAZIONE
   YOLO_MODEL   = yolov10s.pt   # default (puoi usare yolov10m.pt, l, x…)
   YOLO_DEVICE  = cpu           # o "cuda" se il container vede la GPU
   CONF         = 0.4           # soglia confidenza
   VIDEO_URL    = URL IP Camera MJPEG

Dipendenze: ultralytics>=8.2.90, opencv-python-headless, flask
"""

from __future__ import annotations
import os, time
import cv2
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO

# --------- Env config ---------
TARGET       = "person"
YOLO_MODEL   = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_DEVICE  = os.getenv("YOLO_DEVICE", "cpu")
CONF_THRES   = float(os.getenv("CONF", 0.4))
VIDEO_URL    = os.getenv("VIDEO_URL", "http://85.196.146.82:3337/axis-cgi/mjpg/video.cgi")

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")  # sopprime warning Ultralytics

# --------- YOLO init ---------
model = YOLO(YOLO_MODEL, task="detect")
cap   = cv2.VideoCapture(VIDEO_URL)
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire lo stream dalla IP Camera")

prev_t = time.perf_counter()
last_people = 0
last_fps = 0.0
last_ts = time.time()

# --------- Flask app ---------
app = Flask(__name__)


def frame_generator():
    global prev_t, last_people, last_fps, last_ts
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=CONF_THRES, device=YOLO_DEVICE, verbose=False)
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
        last_people, last_fps, last_ts = people, fps, time.time()

        cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/")

def index():
    return render_template_string("""
<!doctype html>
<title>duracamera v10</title>
<style>body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}</style>
<img src="{{ url_for('video_feed') }}" style="max-width:100%;height:auto">
""")


@app.route("/video")

def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/count")

def api_count():
    return jsonify(people=last_people, fps=round(last_fps, 2), ts=int(last_ts))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), threaded=True)
