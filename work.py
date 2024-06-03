from ultralytics import YOLO
from PIL import Image
import cv2

# Load a model
model = YOLO('yolov8n.pt')  # load an official detection model
# model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
# model = YOLO('path/to/best.pt')  # load a custom model

# Track with the model
results = model.predict(source="0", show=True)
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")