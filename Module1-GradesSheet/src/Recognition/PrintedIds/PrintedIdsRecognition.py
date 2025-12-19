import cv2 as cv
import numpy as np
import joblib
import os
import tensorflow as tf
import easyocr
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
METHOD = os.getenv("ID_RECOGNITION_METHOD", "TRADITIONAL")
SVM_PATH = os.getenv("SVM_MODEL_PATH", r'Module1-GradesSheet\models\svm_mnist_model.joblib')
DNN_PATH = os.getenv("DNN_MODEL_PATH", r'Module1-GradesSheet\models\mnist_deep_model.h5')

# Global Model Loading
if METHOD == "DEEP_LEARNING":
    print("Loading Custom DNN...")
    model = tf.keras.models.load_model(DNN_PATH)
elif METHOD == "ALREADY_MADE_OCR":
    print("Initializing Ready-made OCR (EasyOCR)...")
    # This initializes the pretrained DNN for reading text
    reader = easyocr.Reader(['en'], gpu=False) 
else:
    print("Loading Traditional HOG + SVM...")
    model = joblib.load(SVM_PATH)

def extract_hog_features(digit_img):
    """Extract Histogram of Oriented Gradients (HOG) features[cite: 33]."""
    hog = cv.HOGDescriptor(_winSize=(28, 28),
                           _blockSize=(14, 14),
                           _blockStride=(7, 7),
                           _cellSize=(7, 7),
                           _nbins=9)
    return hog.compute(digit_img).flatten()

def segment_digits(cell_img):
    """Adaptive Thresholding + Contour Filtering + Aspect Ratio Splitting."""
    gray = cv.cvtColor(cell_img, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv.THRESH_BINARY_INV, 11, 2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    temp_boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 3 and h > 10:
            temp_boxes.append((x, y, w, h))

    temp_boxes = sorted(temp_boxes, key=lambda b: b[0])
    
    final_digits = []
    for x, y, w, h in temp_boxes:
        aspect_ratio = w / float(h)
        if aspect_ratio > 1.2: 
            half_w = w // 2
            final_digits.append(standardize_digit(binary[y:y+h, x:x+half_w]))
            final_digits.append(standardize_digit(binary[y:y+h, x+half_w:x+w]))
        else:
            final_digits.append(standardize_digit(binary[y:y+h, x:x+w]))
    return final_digits

def standardize_digit(roi):
    h, w = roi.shape
    size = 20
    if h > w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    roi_resized = cv.resize(roi, (new_w, new_h))
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    roi_padded = cv.copyMakeBorder(roi_resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
    return roi_padded

def predict_id_from_cell(cell_img_path):
    if METHOD == "ALREADY_MADE_OCR":
        results = reader.readtext(cell_img_path, allowlist='0123456789')
        return "".join([res[1] for res in results])

    cell_img = cv.imread(cell_img_path)
    if cell_img is None: return "Error"
    
    digits = segment_digits(cell_img)
    predicted_id = ""
    
    for digit_roi in digits:
        if METHOD == "DEEP_LEARNING":
            # (3) Deep Learning Method 
            roi_ready = digit_roi.astype('float32') / 255.0
            roi_ready = np.expand_dims(roi_ready, axis=(0, -1))
            prediction_probs = model.predict(roi_ready, verbose=0)
            prediction = np.argmax(prediction_probs)
        else:
            # (2) Features + Classifier Method 
            features = extract_hog_features(digit_roi).reshape(1, -1)
            prediction = model.predict(features)[0]
        predicted_id += str(prediction)
        
    return predicted_id

# Execution
path = r"Module1-GradesSheet\data\cells\eleven\row2_cell4.jpg"
print(f"User Chosen Method: {METHOD}")
print(f"Predicted ID: {predict_id_from_cell(path)}")