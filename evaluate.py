from ultralytics import YOLO

# Load trained model
model = YOLO(
    "runs/detect/runs/face_mask_detection/weights/best.pt"
)

# Evaluate on validation set
metrics = model.val(
    data="dataset/data.yaml",
    split="val",
    imgsz=640,
    batch=16,
    plots=True
)

print("\n===== VALIDATION RESULTS =====")
print(f"Precision     : {metrics.box.mp:.4f}")
print(f"Recall        : {metrics.box.mr:.4f}")
print(f"mAP50         : {metrics.box.map50:.4f}")
print(f"mAP50-95      : {metrics.box.map:.4f}")

print("\n===== PER-CLASS RESULTS =====")

for class_id, score in enumerate(metrics.box.maps):
    class_name = model.names[class_id]
    print(f"{class_name:<25}: mAP50-95 = {score:.4f}")

print("\nEvaluation completed!")