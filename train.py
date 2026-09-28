from ultralytics import YOLO

# Load pretrained YOLO11 Nano
model = YOLO("yolo11n.pt")

# Train model
results = model.train(
    data="dataset/data.yaml",

    epochs=50,
    imgsz=640,
    batch=16,

    project="runs",
    name="face_mask_detection",

    patience=10,
    workers=4,

    save=True
)


print("Training completed!")
print(
    "Best model: "
    "runs/detect/face_mask_detection/weights/best.pt"
)