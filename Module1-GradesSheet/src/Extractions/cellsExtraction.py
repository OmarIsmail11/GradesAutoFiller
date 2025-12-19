import cv2 as cv
import numpy as np
import os
from utils import *
from tableExtraction import *

# ---------------------- FUNCTIONS ----------------------

def detect_table_cells(image, min_cell_width=30, min_cell_height=20):
    """Detect table cells and group them into rows."""
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Thresholding and invert (lines = white)
    _, img_bin = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    img_bin = cv.bitwise_not(img_bin)

    # Detect vertical lines
    kernel_length_v = img_gray.shape[1] // 120
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length_v))
    temp_img_v = cv.erode(img_bin, vertical_kernel, iterations=5)
    vertical_lines = cv.dilate(temp_img_v, vertical_kernel, iterations=5)

    # Detect horizontal lines
    kernel_length_h = img_gray.shape[0] // 40
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length_h, 1))
    temp_img_h = cv.erode(img_bin, horizontal_kernel, iterations=5)
    horizontal_lines = cv.dilate(temp_img_h, horizontal_kernel, iterations=5)

    # Combine lines to get table skeleton
    table_segment = cv.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    table_segment = cv.erode(cv.bitwise_not(table_segment), kernel, iterations=2)
    _, table_segment = cv.threshold(table_segment, 0, 255, cv.THRESH_OTSU)

    # Find contours
    contours, _ = cv.findContours(table_segment, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter valid cells
    cells = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w >= min_cell_width and h >= min_cell_height:
            cells.append((x, y, w, h))

    # Sort cells top → bottom
    cells = sorted(cells, key=lambda b: b[1])

    # Group cells into rows based on y-coordinate
    rows = []
    current_row = [cells[0]]
    for cell in cells[1:]:
        if abs(cell[1] - current_row[-1][1]) < 20:  # tolerance for same row
            current_row.append(cell)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [cell]
    rows.append(sorted(current_row, key=lambda b: b[0]))

    return rows

def visualize_cells(image, rows):
    """Draw rectangles around each detected cell."""
    output_img = image.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    for i, row in enumerate(rows):
        for (x, y, w, h) in row:
            cv.rectangle(output_img, (x, y), (x+w, y+h), colors[i % 4], 2)
    return output_img

def save_cells(image, rows, output_root="../data/cells", image_index=0):
    """Save each cell into folders per image."""
    folder_path = os.path.join(output_root, str(image_index+1))
    os.makedirs(folder_path, exist_ok=True)

    for i, row in enumerate(rows):
        for j, (x, y, w, h) in enumerate(row):
            cell_img = image[y:y+h, x:x+w]
            cell_path = os.path.join(folder_path, f"row{i+1}_cell{j+1}.jpg")
            cv.imwrite(cell_path, cell_img)

# ---------------------- TEST FUNCTION ----------------------

def test_cell_extraction(image_paths, output_root="../data/cells"):
    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}")
        image = cv.imread(img_path)
        image = extractTable(img_path)
        rows = detect_table_cells(image)
        output_img = visualize_cells(image, rows)
        show_images([output_img], titles=[f"Cells Highlighted - Image {i+1}"])
        save_cells(image, rows, output_root, image_index=i)
        print(f"Saved {len(rows)} rows for image {i+1}")

# ---------------------- USAGE ----------------------

images = ["../data/images/1.jpg", "../data/images/2.jpg", "../data/images/3.jpg", "../data/images/4.jpg",
          "../data/images/5.jpg", "../data/images/6.jpg", "../data/images/7.jpg", "../data/images/8.jpg",
          "../data/images/9.jpg", "../data/images/10.jpg", "../data/images/11.jpg", "../data/images/12.jpg",
          "../data/images/13.jpg", "../data/images/14.jpg", "../data/images/15.jpg", "../data/images/16.jpg",
          "../data/images/17.jpg", "../data/images/18.jpg", "../data/images/19.jpg", "../data/images/20.jpg",
          "../data/images/21.jpg", "../data/images/22.jpg", "../data/images/23.jpg", "../data/images/24.jpg"]

test_cell_extraction(images)
