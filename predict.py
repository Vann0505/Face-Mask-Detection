from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/face_mask_detection/weights/best.pt"
IMAGE_PATH = "test.png"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.5,
    imgsz=640,
    save=True,
    project="C:/Coding/VisMask/outputs",
    name="prediction"
)

# Ambil hasil gambar pertama
result = results[0]

print("\n===== DETECTION RESULTS =====")

# Cek apakah ada objek yang terdeteksi
if result.boxes is None or len(result.boxes) == 0:
    print("No objects detected.")
else:
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"{i + 1}. {class_name} "
            f"- {confidence * 100:.2f}%"
        )

print("\nResult saved to:")
print("C:/Coding/VisMask/outputs/prediction/")