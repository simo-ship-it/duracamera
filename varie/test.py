import cv2

for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_FFMPEG]:
    print(f"--- Test backend {backend}")
    for i in range(4):                 # prova i primi 4 indici
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            print(f"  ✅ Camera index {i} OK con backend {backend}")
            cap.release()
        else:
            print(f"  ❌ Camera index {i} KO")
