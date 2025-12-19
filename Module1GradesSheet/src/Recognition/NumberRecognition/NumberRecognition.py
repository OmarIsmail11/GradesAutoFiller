import cv2 as cv
import numpy as np
import joblib
import os
import tensorflow as tf
import easyocr
from dotenv import load_dotenv

load_dotenv()
SVM_PATH = os.getenv("SVM_MODEL_PATH", r'Module1GradesSheet\models\svm_mnist_model.joblib')
DNN_PATH = os.getenv("DNN_MODEL_PATH", r'Module1GradesSheet\models\mnist_deep_model.h5')

class NumberRecognizer:
    def __init__(self):
        """Pre-loads all available numeric recognition models into a registry."""
        self.models = {}
        self.reader = None
        self._init_all_models()

    def _init_all_models(self):
        # 1. Load Custom CNN (DNN)
        if os.path.exists(DNN_PATH):
            print(f"--- Loading Custom CNN: {DNN_PATH} ---")
            self.models['DEEP_LEARNING'] = tf.keras.models.load_model(DNN_PATH)
            # Warm-up inference
            self.models['DEEP_LEARNING'].predict(np.zeros((1, 28, 28, 1)), verbose=0)
        
        # 2. Load SVM
        if os.path.exists(SVM_PATH):
            print(f"--- Loading SVM: {SVM_PATH} ---")
            self.models['TRADITIONAL'] = joblib.load(SVM_PATH)
            
        # 3. Initialize EasyOCR (Already-made)
        print("--- Initializing EasyOCR for numeric digits ---")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_hog_features(self, digit_img):
        hog = cv.HOGDescriptor(_winSize=(28, 28), _blockSize=(14, 14), 
                               _blockStride=(7, 7), _cellSize=(7, 7), _nbins=9)
        return hog.compute(digit_img).flatten()

    def segment_digits_raw(self, cell_img):
        """Returns raw binary ROIs with their X-coordinates."""
        gray = cv.cvtColor(cell_img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        # Initial cleaning
        kernel = np.ones((2,2), np.uint8)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        initial_blobs = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w > 2 and h > 8: 
                initial_blobs.append({'roi': binary[y:y+h, x:x+w], 'x': x, 'w': w, 'h': h})
        
        # Sort left-to-right
        initial_blobs.sort(key=lambda b: b['x'])
        
        processed_blobs = []
        for blob in initial_blobs:
            roi, x, w, h = blob['roi'], blob['x'], blob['w'], blob['h']
            
            # Use projection to check for internal touching digits
            if (w / float(h)) > 1.1: 
                split_indices = self._get_split_points(roi)
                if len(split_indices) > 2:
                    for i in range(len(split_indices) - 1):
                        x_start = split_indices[i]
                        x_end = split_indices[i+1]
                        if (x_end - x_start) > 4:
                            # Calculate new global X coordinate
                            processed_blobs.append({'roi': roi[:, x_start:x_end], 'x': x + x_start})
                    continue
            
            processed_blobs.append({'roi': roi, 'x': x})
                
        return processed_blobs

    def _get_split_points(self, roi):
        """Finds only significant valleys to prevent over-segmenting single wide digits."""
        projection = np.sum(roi, axis=0) / 255
        width = roi.shape[1]
        
        # Calculate mean density to identify what a 'deep' valley looks like
        avg_density = np.mean(projection)
        
        split_points = [0]
        i = 1
        while i < width - 1:
            # A split point must be:
            # 1. A local minimum
            # 2. Significantly lower than the average density (the 'valley' rule)
            if projection[i] < projection[i-1] and projection[i] < projection[i+1]:
                if projection[i] < (avg_density * 0.4): # Valley must be deep
                    split_points.append(i)
                    i += 5
            i += 1
            
        split_points.append(width)
        return split_points
        
    def standardize_digit(self, roi):
        h, w = roi.shape
        size = 20
        new_h, new_w = (size, int(w * size / h)) if h > w else (int(h * size / w), size)
        roi_resized = cv.resize(roi, (new_w, new_h))
        top, left = (28 - new_h) // 2, (28 - new_w) // 2
        bottom, right = 28 - new_h - top, 28 - new_w - left
        return cv.copyMakeBorder(roi_resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)

    def predict_id_from_cell(self, cell_img, method="TRADITIONAL", expected_digits=None):
        if cell_img is None: return "Error"

        if method == "ALREADY_MADE_OCR":
            results = self.reader.readtext(cell_img, allowlist='0123456789', detail=0)
            
            predicted_id = "".join(results)
            
            if expected_digits and len(predicted_id) != expected_digits:
                print(f"Warning: OCR found {len(predicted_id)} digits, expected {expected_digits}")
                
            return predicted_id
        
        blobs = self.segment_digits_raw(cell_img)

        if expected_digits is not None:
            if len(blobs) > expected_digits:
                blobs.sort(key=lambda b: np.sum(b['roi']), reverse=True)
                blobs = blobs[:expected_digits]
                blobs.sort(key=lambda b: b['x'])
            
            elif len(blobs) < expected_digits:
                while len(blobs) < expected_digits:
                    # Find the blob with the largest width
                    widest_idx = np.argmax([b['roi'].shape[1] for b in blobs])
                    target = blobs.pop(widest_idx)
                    
                    roi, start_x = target['roi'], target['x']
                    mid = roi.shape[1] // 2
                    
                    # Split and insert back while maintaining X order
                    blobs.insert(widest_idx, {'roi': roi[:, mid:], 'x': start_x + mid})
                    blobs.insert(widest_idx, {'roi': roi[:, :mid], 'x': start_x})
                    blobs.sort(key=lambda b: b['x'])

        predicted_id = ""
        model = self.models.get(method)
        
        for b in blobs:
            digit_roi = self.standardize_digit(b['roi'])
            
            if method == "DEEP_LEARNING":
                roi_ready = digit_roi.astype('float32').reshape(1, 28, 28, 1) / 255.0
                prediction = np.argmax(model.predict(roi_ready, verbose=0))
            else:
                features = self.extract_hog_features(digit_roi).reshape(1, -1)
                prediction = model.predict(features)[0]
            predicted_id += str(prediction)
            
        return predicted_id

number_recognizer = NumberRecognizer()