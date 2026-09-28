import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)

MODEL_PATH = "runs/detect/runs/face_mask_detection/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("Face Mask Detection")

st.write(
    "Detect face mask usage using a YOLO11"
)

st.divider()

mode = st.radio(
    "Choose Detection Mode",
    ["🖼️ Upload Image", "📷 Webcam"],
    horizontal=True
)

if mode == "🖼️ Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as temp_file:

            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            conf=0.5,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        # Annotated image
        annotated_image = result.plot()

        st.subheader("Detection Result")

        st.image(
            annotated_image,
            channels="BGR",
            caption="YOLO11 Detection",
            use_container_width=True
        )

        st.subheader("Detection Summary")

        if result.boxes is None or len(result.boxes) == 0:

            st.warning(
                "No face mask-related objects detected."
            )

        else:

            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence
                    }
                )

            # Individual detections
            for i, detection in enumerate(detections):

                st.write(
                    f"**{i + 1}. {detection['class']}** — "
                    f"{detection['confidence'] * 100:.2f}%"
                )

            with_mask = sum(
                1
                for d in detections
                if d["class"] == "with_mask"
            )

            incorrect_mask = sum(
                1
                for d in detections
                if d["class"] == "mask_weared_incorrect"
            )

            without_mask = sum(
                1
                for d in detections
                if d["class"] == "without_mask"
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "With Mask",
                    with_mask
                )

            with col2:
                st.metric(
                    "Incorrect Mask",
                    incorrect_mask
                )

            with col3:
                st.metric(
                    "Without Mask",
                    without_mask
                )

        os.remove(temp_path)

else:

    st.subheader("📷 Webcam Detection")

    st.write(
        "Use your webcam to capture an image and detect face masks."
    )

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(camera_image)

        st.subheader("Captured Image")

        st.image(
            image,
            use_container_width=True
        )

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as temp_file:

            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            conf=0.5,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        annotated_image = result.plot()

        st.subheader("Detection Result")

        st.image(
            annotated_image,
            channels="BGR",
            caption="YOLO11 Detection",
            use_container_width=True
        )

        st.subheader("Detection Summary")

        if result.boxes is None or len(result.boxes) == 0:

            st.warning(
                "No face mask-related objects detected."
            )

        else:

            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence
                    }
                )

            for i, detection in enumerate(detections):

                st.write(
                    f"**{i + 1}. {detection['class']}** — "
                    f"{detection['confidence'] * 100:.2f}%"
                )

            with_mask = sum(
                1
                for d in detections
                if d["class"] == "with_mask"
            )

            incorrect_mask = sum(
                1
                for d in detections
                if d["class"] == "mask_weared_incorrect"
            )

            without_mask = sum(
                1
                for d in detections
                if d["class"] == "without_mask"
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "With Mask",
                    with_mask
                )

            with col2:
                st.metric(
                    "Incorrect Mask",
                    incorrect_mask
                )

            with col3:
                st.metric(
                    "Without Mask",
                    without_mask
                )

        os.remove(temp_path)