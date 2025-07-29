import time, cv2
from ultralytics import YOLO

model = YOLO("yolov10s.pt")
TARGET = "person"

# ──────────────────────────────────────────────────────────────
#  CAM SOURCE: MJPEG via HTTP
#  Nota: OpenCV usa FFMPEG in automatico; se servisse, forza backend:
#  cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
# ──────────────────────────────────────────────────────────────
URL = "https://webcam.sparkassenplatz.info/cgi-bin/faststream.jpg?stream=full&fps=25"
cap = cv2.VideoCapture(URL)                    # oppure CAP_FFMPEG

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire lo stream: {URL}")

prev_t = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ Frame mancante – riconnessione…")
        cap.open(URL)           # tenta di riconnettersi
        continue

    # YOLO inference (CPU)
    results = model.predict(frame, conf=0.4, device="cpu", verbose=False)
    dets = results[0].boxes

    people = 0
    for box in dets:
        if model.names[int(box.cls[0])] == TARGET:
            people += 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FPS overlay
    now = time.perf_counter()
    fps = 1.0 / (now - prev_t)
    prev_t = now
    cv2.putText(frame, f"Persone: {people}  FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counter (MJPEG)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
