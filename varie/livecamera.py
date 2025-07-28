#!/usr/bin/env python3
"""
people_counter_youtube.py
Conta le persone in un video / live YouTube con YOLOv8 e mostra FPS.
"""

import time
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube     # pip install cap_from_youtube yt-dlp

# ─────────────── PARAMETRI -----------------------------------------------------
YOUTUBE_URL = "https://www.youtube.com/watch?v=Z49UkOi08DE"   # sostituisci come vuoi
RESOLUTION  = "480p"          # 144p, 240p, 360p, 480p, 720p, 1080p60, "best"
CONF_THRES  = 0.4              # soglia confidenza YOLO
TARGET      = "person"         # classe COCO da contare
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Apri lo stream YouTube
    cap = cap_from_youtube(YOUTUBE_URL, resolution=RESOLUTION)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire lo stream YouTube")

    # 2. Carica YOLO
    model = YOLO("yolov8n.pt")      # nano; cambia in yolov8s.pt se hai GPU
    prev_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Frame mancante, provo a riconnettere…")
            cap.open(YOUTUBE_URL)   # tenta nuova URL (cap_from_youtube rigenera)
            continue

        # 3. Inferenza
        results = model.predict(frame, conf=CONF_THRES, device="cpu", verbose=False)
        dets = results[0].boxes

        # 4. Conta persone + disegna box
        people = 0
        for box in dets:
            if model.names[int(box.cls[0])] == TARGET:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5. Calcola FPS
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now

        # 6. Overlay testo
        cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 7. Mostra finestra
        cv2.imshow("People Counter – YouTube", frame)
        if cv2.waitKey(1) & 0xFF == 27:     # ESC per uscire
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()