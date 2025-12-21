import cv2
import numpy as np
from typing import Tuple, Dict

class SymbolClassifier:
    """Classical image processing approach to classify hand-drawn symbols"""
    
    def __init__(self):
        self.symbol_types = {
            'checkmark': 'Check Mark',
            'question': 'Question Mark',
            'horizontal_lines': 'Horizontal Lines',
            'vertical_lines': 'Vertical Lines',
            'box': 'Box'
        }
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the image for analysis"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Denoise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        return gray, binary
    
    def extract_features(self, binary: np.ndarray) -> Dict:
        """Extract features from the binary image"""
        features = {}
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return features
        
        # Get the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Basic shape features
        features['num_contours'] = len(contours)
        features['area'] = cv2.contourArea(main_contour)
        features['perimeter'] = cv2.arcLength(main_contour, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0
        features['extent'] = features['area'] / (w * h) if w * h > 0 else 0
        
        # Approximate polygon
        epsilon = 0.02 * features['perimeter']
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        features['vertices'] = len(approx)
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0
        
        # Line detection using HoughLinesP
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=30, 
                                minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            features['num_lines'] = len(lines)
            
            # Classify lines as horizontal or vertical
            horizontal = 0
            vertical = 0
            diagonal = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    horizontal += 1
                elif 70 < angle < 110:
                    vertical += 1
                else:
                    diagonal += 1
            
            features['horizontal_lines'] = horizontal
            features['vertical_lines'] = vertical
            features['diagonal_lines'] = diagonal
        else:
            features['num_lines'] = 0
            features['horizontal_lines'] = 0
            features['vertical_lines'] = 0
            features['diagonal_lines'] = 0
        
        # Moments for shape analysis
        moments = cv2.moments(main_contour)
        if moments['m00'] != 0:
            features['hu_moments'] = cv2.HuMoments(moments).flatten()
        
        return features
    
    def classify(self, image: np.ndarray) -> Tuple[str, int, Dict]:
        """Classify the symbol in the image"""
        gray, binary = self.preprocess(image)
        features = self.extract_features(binary)
        
        if not features:
            return "Unknown", 0, features
        
        # Classification logic based on features
        symbol_type = "Unknown"
        count = 0
        
        # Check for box (rectangle with 4 corners, high solidity)
        if (features.get('vertices', 0) == 4 and 
            features.get('solidity', 0) < 0.3 and
            0.7 < features.get('aspect_ratio', 0) < 1.5):
            symbol_type = 'box'
            count = 1
        
        # Check for horizontal lines (multiple separated horizontal strokes)
        elif (features.get('horizontal_lines', 0) >= 2 and
              features.get('vertical_lines', 0) <= 1 and
              features.get('num_contours', 0) >= 2):
            symbol_type = 'horizontal_lines'
            count = min(features.get('num_contours', 0), 5)
        
        # Check for vertical lines (multiple separated vertical strokes)
        elif (features.get('vertical_lines', 0) >= 2 and
              features.get('horizontal_lines', 0) <= 1 and
              features.get('num_contours', 0) >= 2):
            symbol_type = 'vertical_lines'
            count = min(features.get('num_contours', 0), 5)
        
        # Check for question mark (curved with a dot, or single complex contour)
        elif (features.get('num_contours', 0) >= 1 and
              features.get('solidity', 0) < 0.7 and
              features.get('aspect_ratio', 0) < 0.8):
            # Question marks are typically taller than wide
            symbol_type = 'question'
            count = 1
        
        # Check for checkmark (diagonal line with specific shape)
        elif (features.get('diagonal_lines', 0) >= 1 and
              features.get('num_contours', 0) == 1 and
              features.get('vertices', 0) <= 6):
            symbol_type = 'checkmark'
            count = 1
        
        return symbol_type, count, features
    
    def predict(self, image_path: str) -> Dict:
        """Main prediction function"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'error': 'Could not read image',
                'symbol': 'Unknown',
                'count': 0
            }
        
        symbol_type, count, features = self.classify(image)
        
        result = {
            'symbol': self.symbol_types.get(symbol_type, 'Unknown'),
            'symbol_code': symbol_type,
            'count': count,
            'features': {
                'num_contours': features.get('num_contours', 0),
                'aspect_ratio': round(features.get('aspect_ratio', 0), 2),
                'solidity': round(features.get('solidity', 0), 2),
                'horizontal_lines': features.get('horizontal_lines', 0),
                'vertical_lines': features.get('vertical_lines', 0),
                'diagonal_lines': features.get('diagonal_lines', 0)
            }
        }
        
        return result


# Example usage
if __name__ == "__main__":
    classifier = SymbolClassifier()
    
    # Test with your images
    test_images = [
        'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/cells/12/row4_cell5.jpg',
    ]
    
    for img_path in test_images:
        try:
            result = classifier.predict(img_path)
            print(f"/nImage: {img_path}")
            print(f"Predicted: {result['symbol']}")
            if result['count'] > 0:
                print(f"Count: {result['count']}")
            print(f"Features: {result['features']}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")