#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
durastream – YOLOv10 people-counter con streaming MJPEG/YouTube.
Ottimizzato per latenza bassa in container Docker.
"""

from __future__ import annotations
import os, sys, time, json, atexit, threading, collections, contextlib
from collections import deque

import cv2
import torch
from flask import Flask, Response, jsonify, render_template_string
from ultralytics import YOLO
from yt_dlp import YoutubeDL

# ────────────────────── CONFIGURAZIONE via env ──────────────────────
TARGET        = os.getenv("TARGET", "person")
YOLO_MODEL    = os.getenv("YOLO_MODEL", "yolov10n.pt")      # nano = +veloce
YOLO_DEVICE   = os.getenv("YOLO_DEVICE", "auto")            # auto / cuda / mps / cpu
CONF_THRES    = float(os.getenv("CONF", 0.4))
IMG_SIZE      = int(os.getenv("IMG_SIZE", 640))             # lato lungo in pixel
VIDEO_URL     = os.getenv("VIDEO_URL",
                          "http://192.168.1.67:8080/video")  # YouTube o MJPEG/RTSP
YT_FORMAT     = os.getenv("YT_FORMAT",
                          "best[ext=mp4][height<=720]/best")
RETRY_SEC     = int(os.getenv("RETRY_SEC", 2))              # back-off riconnessione

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")             # sopprime warning Ultralytics
# ────────────────────────────────────────────────────────────────────


# ───────────── Helper: device auto-detect ─────────────
def pick_device(pref: str = "auto") -> str:
    """Restituisce il dispositivo migliore disponibile."""
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_mps_available():
        return "mps"
    return "cpu"


# ───────────── Helper: YouTube via yt-dlp ─────────────
def _pick_stream_url(info: dict) -> str | None:
    if info.get("url") and info.get("ext") == "mp4":
        return info["url"]
    fmts = [
        f for f in info.get("formats", [])
        if f.get("vcodec") != "none" and f.get("protocol", "").startswith("https")
    ]
    fmts.sort(key=lambda f: f.get("height", 0), reverse=True)
    return fmts[0]["url"] if fmts else None


def cap_from_youtube(url: str) -> cv2.VideoCapture:
    with YoutubeDL({"format": YT_FORMAT, "quiet": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        stream = _pick_stream_url(info)
        if not stream:
            sys.stderr.write("[durastream] formato video non compatibile\n")
            raise RuntimeError("Nessun formato video compatibile")
    cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire lo stream YouTube")
    return cap


# ───────────── Helper: sorgenti varie ─────────────
def cap_from_source(url: str) -> cv2.VideoCapture:
    if "youtu" in url:
        return cap_from_youtube(url)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # azzera il buffer ffmpeg
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire lo stream: {url}")
    return cap


# ───────────── YOLO init ─────────────
device = pick_device(YOLO_DEVICE)
print("🖥  Device usato per l’inferenza:", device)
model = YOLO(YOLO_MODEL, task="detect")

# ───────────── Capture & grabbing thread ─────────────
cap = cap_from_source(VIDEO_URL)
atexit.register(lambda: cap.release())

frames: deque = deque(maxlen=1)     # contiene SEMPRE solo l’ultimo frame


def grab_frames():
    global cap
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(RETRY_SEC)
            try:
                new_cap = cap_from_source(VIDEO_URL)
            except Exception as e:
                sys.stderr.write(f"[durastream] reconnect failed: {e}\n")
                continue
            cap.release()
            cap = new_cap
            continue

        frames.append(frame)


threading.Thread(target=grab_frames, daemon=True).start()

# statistiche per endpoint /api/count
fps_win = collections.deque(maxlen=30)
last_people = 0
last_fps = 0.0
last_ts = time.time()
prev_t = time.perf_counter()

# ───────────── Flask app ─────────────
app = Flask(__name__)


def frame_generator():
    global prev_t, last_people, last_fps, last_ts
    while True:
        if not frames:
            time.sleep(0.001)
            continue

        # usa SEMPRE il frame più recente
        frame = frames[-1]

        # ridimensiona lato lungo = IMG_SIZE
        h, w = frame.shape[:2]
        scale = IMG_SIZE / max(h, w)
        frame_proc = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame

        # YOLO inference
        results = model.predict(
            frame_proc,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=device,
            verbose=False,
        )
        dets = results[0].boxes

        people = 0
        for box in dets:
            if model.names[int(box.cls[0])] == TARGET:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS & overlay
        now = time.perf_counter()
        inst_fps = 1.0 / (now - prev_t)
        prev_t = now
        fps_win.append(inst_fps)
        fps = sum(fps_win) / len(fps_win)

        last_people, last_fps, last_ts = people, fps, time.time()

        cv2.putText(frame_proc, f"Persone: {people}  FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buf = cv2.imencode(".jpg", frame_proc)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template_string("""
<!doctype html>
<title>durastream v11</title>
<style>
  body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}
</style>
<img src="{{ url_for('video_feed') }}" style="max-width:100%;height:auto">
""")


@app.route("/video")
def video_feed():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/count")
def api_count():
    return jsonify(people=last_people, fps=round(last_fps, 2), ts=int(last_ts))


if __name__ == "__main__":
    # threaded=True permette più client contemporanei
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), threaded=True)