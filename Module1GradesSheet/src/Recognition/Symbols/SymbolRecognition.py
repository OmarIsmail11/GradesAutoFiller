import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

class SymbolClassifier:
    def __init__(self):
        pass

    def preprocess(self, img):
        """
        Aggressive cleaning for blurry, blue-tinted spreadsheet cells.
        """
        # 1. Grayscale
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. Contrast Enhancement (CLAHE) - Crucial for faint lines
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # 3. Denoise
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # 4. Adaptive Thresholding (Inverted: White symbols on black)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv.THRESH_BINARY_INV, 19, 4)

        # 5. Morphological Closing (Connect broken lines/box edges)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv.CC_STAT_AREA] < 20: # Filter tiny dots
                binary[labels == i] = 0

        return binary

    def count_lines_projection(self, binary_img, axis=0):
        """
        Counts peaks in pixel density to find number of lines.
        axis=0 -> Vertical Projection (counts Vertical lines)
        axis=1 -> Horizontal Projection (counts Horizontal lines)
        """
        # Sum white pixels along the axis
        projection = np.sum(binary_img, axis=axis)
        
        # Normalize
        projection = projection / 255
        
        # Find peaks (lines)
        # distance=5 prevents counting thick lines as double
        # height=5 ensures we don't count noise
        peaks, _ = find_peaks(projection, distance=10, height=5)
        
        return len(peaks)

    def analyze_contours(self, binary, original):
        """
        Main Classification Logic
        """
        # Find Contours with Hierarchy (to detect boxes with holes)
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "Empty"

        # Get the largest parent contour (ignore internal noise)
        # Hierarchy format: [Next, Previous, First_Child, Parent]
        # We look for contours with Parent == -1 (Top level)
        parent_indices = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
        
        if not parent_indices:
            return "Empty"

        # Sort by area, take largest
        largest_idx = max(parent_indices, key=lambda i: cv.contourArea(contours[i]))
        c = contours[largest_idx]
        
        # --- GEOMETRIC FEATURES ---
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv.contourArea(c)
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if hull_area == 0: hull_area = 1
        solidity = area / hull_area
        
        # Check for holes (Child contours)
        # If hierarchy[0][largest_idx][2] != -1, it has a child -> Likely a Box
        has_hole = hierarchy[0][largest_idx][2] != -1

        # --- LOGIC TREE ---

        # 1. CHECK FOR MULTIPLE LINES (Projection Method)
        # This is more robust than contour counting for broken lines
        v_lines = self.count_lines_projection(binary, axis=0) # Sum columns
        h_lines = self.count_lines_projection(binary, axis=1) # Sum rows

        # If we see clearly separated lines
        if v_lines > 1 and aspect_ratio < 2.0: # AR check prevents confusing a Box for 2 lines
             return f"{v_lines} Vertical Lines"
        
        if h_lines > 1 and aspect_ratio > 0.5:
             return f"{h_lines} Horizontal Lines"

        # 2. CHECK FOR BOX
        # A box usually has a hole (Euler number) OR roughly 4 corners
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        
        if has_hole:
            return "Box"
        
        if len(approx) == 4 and 0.8 < aspect_ratio < 1.2 and solidity > 0.7:
             return "Box"

        # 3. CHECK FOR QUESTION MARK
        # Question marks are 'Top Heavy' and usually consist of 2 parts (body + dot)
        # or have a very specific aspect ratio (Tall) with low solidity (curve)
        
        # Calculate Center of Mass
        M = cv.moments(c)
        if M["m00"] != 0:
            cY = int(M["m01"] / M["m00"])
            # Normalized center Y (0 is top, 1 is bottom)
            norm_cY = (cY - y) / h
        else:
            norm_cY = 0.5

        # Logic: 
        # - Tall (AR < 0.6)
        # - Top heavy (norm_cY < 0.5) implies the 'hook' is the main mass
        # - OR: We found 2 distinct contours arranged vertically
        if len(parent_indices) == 2:
            # Sort vertically
            sorted_cnts = sorted([contours[i] for i in parent_indices], key=lambda b: cv.boundingRect(b)[1])
            # Check if bottom one is small (dot)
            _, _, w_dot, h_dot = cv.boundingRect(sorted_cnts[1])
            if (w_dot * h_dot) < (area * 0.3):
                return "Question Mark"

        if aspect_ratio < 0.8 and norm_cY < 0.45:
             return "Question Mark"

        # 4. CHECK FOR CHECK MARK
        # - Not a box, not a line.
        # - Aspect ratio is moderate (0.8 - 1.5)
        # - Low solidity (The "V" shape creates a big empty convex hull)
        if solidity < 0.7:
            return "Check Mark"
            
        # 5. FALLBACK: SINGLE LINES
        if aspect_ratio > 3.0:
            return "1 Horizontal Line"
        if aspect_ratio < 0.3:
            return "1 Vertical Line"

        # Default fallback
        return "Unknown Shape (Likely Check Mark or Noise)"

    def predict(self, image_path):
        img = cv.imread(image_path)
        if img is None: return "Error loading image"
        
        processed = self.preprocess(img)
        result = self.analyze_contours(processed, img)
        return result

# --- USAGE ---
classifier = SymbolClassifier()

# List of your files
files = [
    'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/cells/1/row2_cell5.jpg', # Check Mark
    'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/cells/5/row5_cell5.jpg', # Question Mark
    'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/cells/8/row4_cell6.jpg', # Vertical Lines
    'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/cells/1/row8_cell5.jpg', # Horizontal Lines (Blurry)
    'C:/Users/youse/Desktop/University/Image/GradesAutoFiller/Module1GradesSheet/data/cells/1/row10_cell5.jpg', # Box
]

for f in files:
    print(f"File: {f} -> Prediction: {classifier.predict(f)}")