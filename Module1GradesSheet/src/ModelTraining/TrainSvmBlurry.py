import cv2 as cv
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf # Just for loading MNIST easily
import random

# --- CONFIGURATION ---
OUTPUT_MODEL_PATH = 'svm_mnist_blurry.joblib'

def degrade_image(image):
    """
    Simulates the low-quality spreadsheet look:
    1. Gaussian Blur (simulates out-of-focus camera)
    2. Downscale/Upscale (simulates low resolution)
    """
    # 1. Random Blur
    # We use a kernel size of 3 or 5 to match your 'ghostly' digits
    if random.random() > 0.5:
        image = cv.GaussianBlur(image, (3, 3), 1.0)
    else:
        image = cv.GaussianBlur(image, (5, 5), 1.5)
        
    # 2. Downscaling (Pixelation)
    # Shrink to 14x14 or 10x10 and scale back up to destroy fine details
    h, w = image.shape
    scale_factor = random.uniform(0.4, 0.7) # Shrink to 40-70%
    small = cv.resize(image, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv.INTER_NEAREST)
    image = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)
    
    return image

def extract_hog_features(img):
    """
    Must match your production HOG settings EXACTLY.
    """
    # Ensure image is 28x28
    if img.shape != (28, 28):
        img = cv.resize(img, (28, 28))
        
    hog = cv.HOGDescriptor(_winSize=(28, 28), _blockSize=(14, 14), 
                           _blockStride=(7, 7), _cellSize=(7, 7), _nbins=9)
    return hog.compute(img).flatten()

print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Augmenting Data (Making it blurry)...")
x_train_aug = []
y_train_aug = []

# mix: 
# 30% Clean Data
# 70% Blurry Data

for i, img in enumerate(x_train):
    if i % 10000 == 0: print(f"Processing image {i}/{len(x_train)}")
    
    if i < 20000:
        x_train_aug.append(extract_hog_features(img))
        y_train_aug.append(y_train[i])
    else:
        degraded = degrade_image(img)
        x_train_aug.append(extract_hog_features(degraded))
        y_train_aug.append(y_train[i])

x_test_features = [extract_hog_features(degrade_image(img)) for img in x_test]

model = SVC(kernel='rbf', C=5.0, gamma='scale') 
model.fit(x_train_aug, y_train_aug)

# 4. Evaluate
print("Evaluating on Blurry Test Set...")
predictions = model.predict(x_test_features)
acc = accuracy_score(y_test, predictions)
print(f"--- Accuracy on Blurry Data: {acc*100:.2f}% ---")

# 5. Save
print(f"Saving model to {OUTPUT_MODEL_PATH}...")
joblib.dump(model, OUTPUT_MODEL_PATH)
print("Done!")