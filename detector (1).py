from ultralytics import YOLO
import numpy as np
import os

class PPEDetector:
    def __init__(self,model_path:str):
        self.model = YOLO(model_path)
        
    def detect(self, image):
        results = self.model(image, conf=0.20)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return detections


