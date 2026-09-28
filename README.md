Face Mask Detection using YOLO11

A computer vision project for detecting face mask usage using **YOLO11n**. The model identifies three conditions: wearing a mask, wearing a mask incorrectly, and not wearing a mask.
The project includes model training, evaluation, image detection, webcam detection, and a Streamlit web application for interactive inference.

Detection Classes
| Class                   | Description                       |
| ----------------------- | --------------------------------- |
| with_mask               | Person wearing a mask correctly   |
| mask_weared_incorrect   | Person wearing a mask incorrectly |
| without_mask            | Person not wearing a mask         |

Model Performance
The model was trained for 50 epochs using the YOLO11n architecture.
| Metric    |      Score |
| --------- | ---------: |
| Precision | 0.9002     |
| Recall    | 0.7884     |
| mAP@50    | 0.8480     |
| mAP@50-95 | 0.5634     |

Per-Class Performance
| Class                   | mAP@50-95 |
| ----------------------- | --------: |
| mask_weared_incorrect   |    0.4900 |
| with_mask               |    0.6399 |
| without_mask            |    0.5601 |
The evaluation was performed on the validation split because the original dataset does not contain a separate test split.

Dataset
The project uses a Face Mask Detection dataset provided in YOLO format.
Dataset split:
Training: 2,034 images
Validation: 170 images
Total: 2,204 images

The dataset contains three object detection classes:
0 → mask_weared_incorrect
1 → with_mask
2 → without_mask


Tech Stack
* **Python**
* **YOLO11**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **Streamlit**
* **Pillow**
* **NumPy**


Installation
git clone https://github.com/Vann0505/Face-Mask-Detection.git
cd Face-Mask-Detection


Create a virtual environment:
python -m venv venv

Activate the environment on Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt





Training
The model can be trained using:
python train.py

Training configuration:
Model      : YOLO11n
Epochs     : 50
Image Size : 640
Batch Size : 16


The trained model is saved as:
runs/detect/runs/face_mask_detection/weights/best.pt

Evaluation
Run model evaluation with:
python evaluate.py

The evaluation uses the validation dataset and reports:
-Precision
-Recall
-mAP@50
-mAP@50-95
-Per-class mAP@50-95


Image Detection
To run detection on an image, place an image such as:
test.png
in the project directory and run:
python predict.py
The annotated result will be saved in:
outputs/prediction/

Webcam Detection
Run real-time detection using your webcam:
python webcam.py
Press: Q to close the webcam window.

Streamlit Application
Run the interactive web application:
streamlit run app.py


The application provides two detection modes:

*Upload Image
Upload an image and receive:
-Bounding boxes
-Predicted class
-Confidence score
-Detection summary

*Webcam
Capture an image directly using the browser camera and run mask detection.



Example Detection
The model produces detections such as:
with_mask — 90.77%
along with a bounding box around the detected person.

Project Objective
This project was developed to gain practical experience with:
-Object detection
-YOLO model training
-Dataset preparation
-Model evaluation
-Computer vision inference
-Real-time webcam processing
- Building an ML application with Streamlit

It demonstrates an end-to-end computer vision workflow, from dataset preparation and model training to deployment as an interactive application.


Future Improvements
Potential improvements include:
-Training with a larger and more diverse dataset
-Improving detection of incorrectly worn masks
-GPU-accelerated inference
-Real-time browser-based video detection
-Model optimization for faster inference
-Deployment as a public web application
