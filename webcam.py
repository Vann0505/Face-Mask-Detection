import cv2
from ultralytics import YOLO

# Path model
MODEL_PATH = "runs/detect/runs/face_mask_detection/weights/best.pt"

# Load model
model = YOLO(MODEL_PATH)

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read frame.")
        break

    # Resize frame agar inference lebih ringan
    frame = cv2.resize(frame, (640, 480))

    # YOLO prediction
    results = model.predict(
        source=frame,
        conf=0.5,
        imgsz=320,
        verbose=False
    )

    # Box
    annotated_frame = results[0].plot()
    cv2.imshow(
        "Face Mask Detection",
        annotated_frame
    )

    # q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup webcam
cap.release()
cv2.destroyAllWindows()

print("Webcam closed.")