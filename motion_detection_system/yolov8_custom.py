from ultralytics import YOLO
import os

model = YOLO('yolov8m-pose.pt')

img_path = "examples/elderly-cover.jpg"

results = model.predict(source=img_path, show=True, conf=0.5, stream=False, save=True)