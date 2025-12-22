import cv2 as cv
import numpy as np
import joblib
import os
import tensorflow as tf
import easyocr
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# --- MODEL PATHS ---
SVM_PATH = os.getenv("SVM_MODEL_PATH", r'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/svm_mnist_blurry.joblib')
DNN_PATH = os.getenv("DNN_MODEL_PATH", r'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/models/mnist_deep_model.h5')
YOLO_PATH = os.getenv("SEGMENTATION_YOLO_MODEL_PATH", r'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/runs/detect/digit_segmenter4/weights/best.pt')

class NumberRecognizer:
    def __init__(self):
        self.models = {}
        self.reader = None
        self._init_all_models()

    def _init_all_models(self):
        # 1. Load Custom CNN (DNN)
        if os.path.exists(DNN_PATH):
            print(f"--- Loading Custom CNN: {DNN_PATH} ---")
            self.models['DEEP_LEARNING'] = tf.keras.models.load_model(DNN_PATH)
            self.models['DEEP_LEARNING'].predict(np.zeros((1, 28, 28, 1)), verbose=0)
        
        # 2. Load SVM
        if os.path.exists(SVM_PATH):
            print(f"--- Loading SVM: {SVM_PATH} ---")
            self.models['TRADITIONAL'] = joblib.load(SVM_PATH)
            
        # 3. Load YOLO (Segmentation)
        if os.path.exists(YOLO_PATH):
            print(f"--- Loading YOLO Segmenter: {YOLO_PATH} ---")
            self.yolo_model = YOLO(YOLO_PATH)
        else:
            print(f"Warning: YOLO model not found at {YOLO_PATH}")
            self.yolo_model = None

        print("--- Initializing EasyOCR ---")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_hog_features(self, digit_img):
        hog = cv.HOGDescriptor(_winSize=(28, 28), _blockSize=(14, 14), 
                               _blockStride=(7, 7), _cellSize=(7, 7), _nbins=9)
        return hog.compute(digit_img).flatten()

    def segment_digits_raw(self, cell_img, method="YOLO"):
        if method == "YOLO":
            if self.yolo_model is None:
                print("Error: YOLO method requested but model not loaded.")
                return []
            return self._segment_yolo(cell_img)
            
        elif method == "CLASSICAL":
            return self._segment_classical(cell_img)
            
        else:
            print(f"Error: Unknown segmentation method '{method}'")
            return []

    def _segment_yolo(self, cell_img):
        if len(cell_img.shape) == 2:
            img_rgb = cv.cvtColor(cell_img, cv.COLOR_GRAY2RGB)
        else:
            img_rgb = cv.cvtColor(cell_img, cv.COLOR_BGR2RGB)

        # Run Inference
        results = self.yolo_model.predict(img_rgb, conf=0.25, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        
        blobs = []
        h_img, w_img = cell_img.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = box
            # 1px padding
            x1 = max(0, x1 - 1)
            y1 = max(0, y1 - 1)
            x2 = min(w_img, x2 + 1)
            y2 = min(h_img, y2 + 1)
            
            # Crop
            crop = cell_img[y1:y2, x1:x2]
            if len(crop.shape) == 3:
                crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            
            blobs.append({'roi': crop, 'x': x1})

        blobs.sort(key=lambda b: b['x'])
        return blobs

    def _segment_classical(self, cell_img):
        gray = cv.cvtColor(cell_img, cv.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
        
        gray = cv.fastNlMeansDenoising(gray, h=10)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv.THRESH_BINARY_INV, 19, 2)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        
        h_img, w_img = binary.shape
        cv.rectangle(binary, (0, 0), (w_img, h_img), 0, 3)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if h < (h_img * 0.3): continue
            if w < 3: continue
            
            roi = binary[y:y+h, x:x+w]
            aspect_ratio = w / float(h)
            
            # Geometric Splitting
            if aspect_ratio <= 1.2:
                blobs.append({'roi': roi, 'x': x})
            elif 1.2 < aspect_ratio <= 2.2:
                mid = w // 2
                split_x = self._find_best_split_col(roi, search_range=(0.4, 0.6))
                blobs.append({'roi': roi[:, :split_x], 'x': x})
                blobs.append({'roi': roi[:, split_x:], 'x': x + split_x})
            elif aspect_ratio > 2.2:
                third = w // 3
                s1 = self._find_best_split_col(roi[:, :2*third], search_range=(0.4, 0.6))
                s2 = self._find_best_split_col(roi[:, s1:], search_range=(0.4, 0.6)) + s1
                blobs.append({'roi': roi[:, :s1], 'x': x})
                blobs.append({'roi': roi[:, s1:s2], 'x': x + s1})
                blobs.append({'roi': roi[:, s2:], 'x': x + s2})

        blobs.sort(key=lambda b: b['x'])
        return blobs

    def _find_best_split_col(self, roi, search_range=(0.4, 0.6)):
        h, w = roi.shape
        start = int(w * search_range[0])
        end = int(w * search_range[1])
        projection = np.sum(roi, axis=0)
        if end > start:
            min_idx = np.argmin(projection[start:end]) + start
            return min_idx
        return w // 2

    def standardize_digit(self, roi):
        h, w = roi.shape
        if h == 0 or w == 0: return np.zeros((28,28), dtype=np.uint8)

        corners = [roi[0,0], roi[0, w-1], roi[h-1, 0], roi[h-1, w-1]]
        avg_corner = np.mean(corners)

        if avg_corner > 100: 
            roi = cv.bitwise_not(roi)
            
            _, roi = cv.threshold(roi, 50, 255, cv.THRESH_TOZERO)

        roi = cv.normalize(roi, None, 0, 255, cv.NORM_MINMAX)

        max_dim = max(h, w)
        scale = 20.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv.resize(roi, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        delta_w = 28 - new_w
        delta_h = 28 - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        return cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    def predict_id_from_cell(self, cell_img, recognition_method="TRADITIONAL", segmentation_method="YOLO", expected_digits=None):
        if cell_img is None: return "Error"

        if recognition_method == "ALREADY_MADE_OCR":
            if self.reader:
                results = self.reader.readtext(cell_img, allowlist='0123456789', detail=0)
                return "".join(results)
            return "Error: EasyOCR not loaded"
        
        blobs = []
        
        if expected_digits == 1:
            if len(cell_img.shape) == 3:
                gray = cv.cvtColor(cell_img, cv.COLOR_BGR2GRAY)
            else:
                gray = cell_img
            
            blobs = [{'roi': gray, 'x': 0}]
            
        else:
            blobs = self.segment_digits_raw(cell_img, method=segmentation_method)


        if expected_digits is not None:
             if len(blobs) > expected_digits:
                 blobs.sort(key=lambda b: np.sum(b['roi']), reverse=True)
                 blobs = blobs[:expected_digits]
                 blobs.sort(key=lambda b: b['x'])
             
             elif len(blobs) < expected_digits and len(blobs) > 0:
                 print(f"Warning: Found {len(blobs)} segments but expected {expected_digits}. Splitting widest blobs.")
                 
                 while len(blobs) < expected_digits:
                     widths = [b['roi'].shape[1] for b in blobs]
                     widest_idx = np.argmax(widths)
                     target = blobs.pop(widest_idx)
                     
                     roi, start_x = target['roi'], target['x']
                     mid = roi.shape[1] // 2
                     
                     left_blob = {'roi': roi[:, :mid], 'x': start_x}
                     right_blob = {'roi': roi[:, mid:], 'x': start_x + mid}
                     
                     blobs.append(left_blob)
                     blobs.append(right_blob)
                     
                     blobs.sort(key=lambda b: b['x'])

        predicted_id = ""
        model = self.models.get(recognition_method)
        
        if not model:
            return f"Error: Model {recognition_method} not loaded"

        for b in blobs:
            digit_roi = self.standardize_digit(b['roi'])
            
            if recognition_method == "DEEP_LEARNING":
                roi_ready = digit_roi.astype('float32').reshape(1, 28, 28, 1) / 255.0
                prediction = np.argmax(model.predict(roi_ready, verbose=0))
            else:
                features = self.extract_hog_features(digit_roi).reshape(1, -1)
                prediction = model.predict(features)[0]
            
            predicted_id += str(prediction)
            
        return predicted_id

number_recognizer = NumberRecognizer()