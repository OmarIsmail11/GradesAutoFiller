import cv2 as cv
import numpy as np
import joblib
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def extract_hog_features(digit_img):
    hog = cv.HOGDescriptor(_winSize=(28, 28),
                           _blockSize=(14, 14),
                           _blockStride=(7, 7),
                           _cellSize=(7, 7),
                           _nbins=9)
    return hog.compute(digit_img).flatten()

def train_mnist_svm():
    print("Fetching MNIST dataset")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(np.int8)
    
    X_train, y_train = X, y 

    print(f"Extracting HOG features from {len(X_train)} images...")
    hog_features = []
    for img in X_train:
        img_2d = img.reshape(28, 28).astype(np.uint8)
        hog_features.append(extract_hog_features(img_2d))
    
    hog_features = np.array(hog_features)
    
    print("Training SVM on HOG features...")
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(hog_features, y_train)
    
    joblib.dump(clf, 'svm_mnist_model.joblib')
    print("Model saved to svm_mnist_model.joblib")

if __name__ == "__main__":
    train_mnist_svm()