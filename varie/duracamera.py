import time
import cv2
from ultralytics import YOLO

# Modello
model = YOLO("yolov8n.pt")
TARGET = "person"

# Videocamera (sostituisci indice/backend se serve)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la telecamera")

prev_t = time.perf_counter()       # per il primissimo frame

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Inference
    results = model.predict(frame, conf=0.4, device="cpu", verbose=False)
    dets = results[0].boxes

    # Conta persone
    people = 0
    for box in dets:
        if model.names[int(box.cls[0])] == TARGET:
            people += 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calcola FPS
    now = time.perf_counter()
    fps = 1.0 / (now - prev_t)
    prev_t = now

    # Overlay testo
    cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostra finestra
    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC per uscire
        break

cap.release()
cv2.destroyAllWindows()
