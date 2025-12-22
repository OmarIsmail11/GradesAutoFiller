import numpy as np
import os
from ultralytics import YOLO

class YOLOSymbolRecognizer:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"Warning: YOLO Model not found at {model_path}")
            self.model = None
        else:
            print(f"Loading YOLO Symbol Model from: {model_path}")
            self.model = YOLO(model_path)
            
        self.class_names = {
            0: 'Empty',
            1: 'Horizontal',
            2: 'Vertical', 
            3: 'box', 
            4: 'check', 
            5: 'question'
        }

    def predict(self, cellImage, conf=0.5):
        if self.model is None:
            return "Error: YOLO Not Loaded"

        results = self.model.predict(cellImage, conf=conf, verbose=False)[0]
        
        # Count detections
        counts = {name: 0 for name in self.class_names.values()}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.class_names.get(cls_id)
            if label:
                counts[label] += 1
        
        if counts['box'] > 0:
            return 0
        elif counts['question'] > 0:
            return -1
        elif counts['check'] > 0:
            return 5
        elif counts['Vertical'] > 0:
            return min(counts['Vertical'], 5)
        elif counts['Horizontal'] > 0:
            return max(0, 5 - counts['Horizontal'])
        elif counts['Empty'] > 0:
            return ""
        else:
            return ""