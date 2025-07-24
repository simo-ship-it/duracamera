import cv2
from ultralytics import YOLO

# Scegli il modello: 'yolov8n.pt' (più collaudato) o 'yolov9n.pt' (ancora più preciso)
model = YOLO("yolov8n.pt")        # sostituisci con "yolov9n.pt" se preferisci
TARGET = "person"                 # classe COCO da contare

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
         # 0 = webcam; per IP cam usa l'URL RTSP
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la telecamera")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Inference (se hai la GPU, device=0 usa CUDA; -1 forza CPU)
    results = model.predict(frame, conf=0.4, device="cpu", verbose=False)

    dets = results[0].boxes

    people = 0
    for box in dets:
        cls = int(box.cls[0])
        if model.names[cls] == TARGET:
            people += 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Persone: {people}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:       # ESC per uscire
        break

cap.release()
cv2.destroyAllWindows()
