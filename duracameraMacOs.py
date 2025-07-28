#!/usr/bin/env python3
"""People Counter ottimizzato per macOS – versione parametrica

Esempi d'uso:
  # YOLOv10 small, 640 px
  python people_counter_macos.py --weights yolov10s.pt --imgsz 640

  # YOLOv8 nano, 416 px (super-veloce)
  python people_counter_macos.py -w yolov8n.pt -i 416
"""
import argparse
import time
import cv2
import torch
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# ARGOMENTI CLI ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="People counter con YOLO su macOS")
    p.add_argument("--weights", "-w", default="yolov10s.pt",
                   help="Percorso o nome modello Ultralytics (es. yolov10n.pt)")
    p.add_argument("--imgsz", "-i", type=int, default=640,
                   help="Dimensione lato minimo/massimo immagine di input")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence soglia")
    p.add_argument("--device", "-d", default=None,
                   help="Forza device: cpu, mps o cuda. Default = auto")
    return p.parse_args()

args = parse_args()

# -----------------------------------------------------------------------------
# CONFIGURAZIONE DISPOSITIVO ----------------------------------------------------
# -----------------------------------------------------------------------------

auto_device = "mps" if torch.backends.mps.is_available() else "cpu"
device = args.device or auto_device

# -----------------------------------------------------------------------------
# CARICAMENTO MODELLO -----------------------------------------------------------
# -----------------------------------------------------------------------------

model = YOLO(args.weights)
model.fuse()

# Conversione half solo se non siamo su CPU e il backend lo supporta
if device != "cpu":
    model.to(device).half()

# Ricava l'ID della classe "person" dal dict names (nuovo formato Ultralytics)
try:
    PERSON_ID = next(k for k, v in model.names.items() if v == "person")
except StopIteration:
    raise ValueError("La classe 'person' non esiste nel modello selezionato")

# Warm-up (compila kernel)
with torch.no_grad():
    model.predict(
        torch.zeros(
            1,
            3,
            args.imgsz,
            args.imgsz,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device=device,
        ),
        imgsz=args.imgsz,
    )

# -----------------------------------------------------------------------------
# VIDEOCAMERA ------------------------------------------------------------------
# -----------------------------------------------------------------------------

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la telecamera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.imgsz)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.imgsz * 3 / 4))  # 4:3 ratio

prev_t = time.perf_counter()

# -----------------------------------------------------------------------------
# LOOP PRINCIPALE --------------------------------------------------------------
# -----------------------------------------------------------------------------

with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference ------------------------------------------------------------
        result = model.predict(
            frame,
            device=device,
            conf=args.conf,
            imgsz=args.imgsz,
            half=(device != "cpu"),
            verbose=False,
        )[0]

        # Conta persone & disegna bbox ----------------------------------------
        people = 0
        for box in result.boxes:
            if int(box.cls[0]) == PERSON_ID:
                people += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS ------------------------------------------------------------------
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now

        cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Output video ---------------------------------------------------------
        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

cap.release()
cv2.destroyAllWindows()