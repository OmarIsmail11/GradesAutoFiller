import cv2 as cv
import numpy as np
import os
from utils import *
from tableExtraction import *

# -------------------- FUNCTIONS --------------------

def extract_table_lines(image):
    """Extract vertical and horizontal lines from a paper image."""
    grayScale = convertToGrayScale(image)
    _, binary = cv.threshold(grayScale, 127, 255, cv.THRESH_BINARY_INV)

    # Vertical lines
    verticalSE = cv.getStructuringElement(cv.MORPH_RECT, (1, 20))
    erodedImage = cv.erode(binary, verticalSE, iterations=10)
    verticalLines = cv.dilate(erodedImage, verticalSE, iterations=15)

    # Horizontal lines
    horizontalSE = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    erodedImage = cv.erode(binary, horizontalSE, iterations=10)
    horizontalLines = cv.dilate(erodedImage, horizontalSE, iterations=20)

    # Combine lines
    verticalHorizontalLines = cv.add(verticalLines, horizontalLines)

    # Make sure binary
    _, tableMask = cv.threshold(verticalHorizontalLines, 127, 255, cv.THRESH_BINARY)

    return tableMask

def find_table_contours(tableMask):
    """Find contours from table mask."""
    contours, hierarchy = cv.findContours(tableMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def extract_cells_from_contours(contours, minWidth=30, minHeight=20):
    """Extract bounding rectangles of valid cells from contours."""
    cells = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > minWidth and h > minHeight:
            cells.append((x, y, w, h))
    return cells

def group_cells_into_rows(cells, tolerance=20):
    """Group sorted cells into rows based on y-coordinate proximity."""
    cells = sorted(cells, key=lambda b: b[1])
    rows = []
    row = [cells[0]]

    for i in range(1, len(cells)):
        if abs(cells[i][1] - row[-1][1]) < tolerance:
            row.append(cells[i])
        else:
            rows.append(sorted(row, key=lambda b: b[0]))
            row = [cells[i]]
    rows.append(sorted(row, key=lambda b: b[0]))
    return rows

def visualize_rows(image, rows):
    """Draw rectangles for each cell in each row for visualization."""
    rowImage = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    for i, r in enumerate(rows):
        for (x, y, w, h) in r:
            cv.rectangle(rowImage, (x, y), (x+w, y+h), colors[i % 4], 2)
    return rowImage

def process_image(imagePath):
    """Complete pipeline: extract paper, detect table lines, find cells, group rows."""
    paper = extractPaper(imagePath)
    tableMask = extract_table_lines(paper)
    contours = find_table_contours(tableMask)
    cells = extract_cells_from_contours(contours)
    rows = group_cells_into_rows(cells)
    grayScale = convertToGrayScale(paper)
    rowImage = visualize_rows(grayScale, rows)
    return paper, rows, rowImage

def save_cells(imagePath, rows, outputRoot="../data/cells"):
    """Save each detected cell into a folder named after the image."""
    paper = extractPaper(imagePath)
    imageName = os.path.splitext(os.path.basename(imagePath))[0]
    folderPath = os.path.join(outputRoot, imageName)
    os.makedirs(folderPath, exist_ok=True)

    for i, r in enumerate(rows):
        for j, (x, y, w, h) in enumerate(r):
            cellImage = paper[y:y+h, x:x+w]
            outputPath = os.path.join(folderPath, f"row{i+1}_cell{j+1}.jpg")
            cv.imwrite(outputPath, cellImage)

# -------------------- TEST FUNCTION --------------------

def test_table_cell_extraction():
    images = ["../data/images/1.jpg", "../data/images/2.jpg", "../data/images/3.jpg", "../data/images/4.jpg", "../data/images/5.jpg", "../data/images/6.jpg", "../data/images/7.jpg", "../data/images/8.jpg", "../data/images/9.jpg",
              "../data/images/10.jpg", "../data/images/11.jpg", "../data/images/12.jpg", "../data/images/13.jpg", "../data/images/14.jpg", "../data/images/15.jpg", "../data/images/16.jpg", "../data/images/17.jpg", "../data/images/18.jpg",
              "../data/images/19.jpg", "../data/images/20.jpg", "../data/images/21.jpg", "../data/images/22.jpg", "../data/images/23.jpg","../data/images/24.jpg"]

    for i, imgPath in enumerate(images):
        paper, rows, rowImage = process_image(imgPath)
        # Optional visualization
        show_images([paper, rowImage], titles=[f"Original Paper {i+1}", f"Detected Rows {i+1}"])
        save_cells(imgPath, rows)
        print(f"Processed and saved cells for image {i+1}")

# -------------------- MAIN --------------------

if __name__ == "__main__":
    test_table_cell_extraction()
