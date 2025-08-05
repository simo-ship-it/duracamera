#!/usr/bin/env python3
"""
People-counter in tempo reale con YOLOv10 e stream MJPEG (IP Webcam).

• Apple Silicon GPU (MPS) se disponibile; altrimenti CPU
• Modello YOLOv10-nano → ~20 fps @ 640 × 360 su MacBook Air M1
• Lettura frame in un thread con deque(maxlen=1) → zero buffering
• Riconnessione automatica se lo stream cade
"""

import time
import cv2
import threading
from collections import deque
from ultralytics import YOLO
import torch

# ────────────────────────────── CONFIGURAZIONE ──────────────────────────────
URL           = "http://192.168.1.67:8080/video"   # endpoint MJPEG diretto
TARGET_CLASS  = "person"                          # oggetto da contare
MODEL_WEIGHTS = "yolov10n.pt"                     # nano = più veloce
IMG_SIZE      = 640                               # lato lungo d’ingresso
CONF_THRES    = 0.4                               # soglia confidenza
RETRY_SEC     = 2                                 # attesa prima di riconnettere
# ─────────────────────────────────────────────────────────────────────────────

# 1. Selezione device (GPU MPS se c'è)
device = "mps" if torch.backends.mps.is_available() and \
                 torch.backends.mps.is_available() else "cpu"
print(f"🖥  Device selezionato: {device}")

# 2. Caricamento modello
model = YOLO(MODEL_WEIGHTS)

# 3. Apertura stream
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)               # elimina la coda FFmpeg
if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire lo stream: {URL}")

# 4. Thread di grabbing non-bloccante
frames = deque(maxlen=1)

def grab_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Nessun frame – riconnessione fra", RETRY_SEC, "s")
            time.sleep(RETRY_SEC)
            cap.open(URL)
            continue
        frames.append(frame)

threading.Thread(target=grab_frames, daemon=True).start()

# 5. Loop principale: inferenza + overlay
prev_t = time.perf_counter()

try:
    while True:
        if not frames:
            time.sleep(0.001)          # evita busy-wait
            continue

        frame = frames[-1]

        # Riduci risoluzione: 640 × H mantiene aspect ratio
        h, w = frame.shape[:2]
        scale = IMG_SIZE / max(h, w)
        if scale < 1:                                   # riduci solo se serve
            frame_res = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            frame_res = frame

        # YOLOv10 inference
        results = model.predict(
            frame_res,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=device,
            verbose=False
        )
        dets = results[0].boxes

        # Conta persone + disegna bounding box
        people = 0
        for box in dets:
            cls_name = model.names[int(box.cls[0])]
            if cls_name == TARGET_CLASS:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame_res, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame_res, f"Persone: {people}  FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("People Counter (MJPEG)", frame_res)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):     # ESC o q per uscire
            break

finally:
    cap.release()
    cv2.destroyAllWindows()