import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from Module1GradesSheet.src.utils import *
from Module1GradesSheet.src.Extractions.paperExtraction import *

def readImage(imagePath):
    # Read an image and convert to RGB
    image = cv.imread(imagePath)
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return imageRGB

def getPaperImage(imagePath):
    # Extract the paper from an image
    paperImage = extractPaper(imagePath)
    return paperImage

def threshold(paperImage):
    # Convert paper to grayscale and apply adaptive thresholding
    paperGrayScale = cv.cvtColor(paperImage, cv.COLOR_RGB2GRAY)
    gaussian = cv.GaussianBlur(paperGrayScale, (7,7), 0)
    binary = cv.adaptiveThreshold(gaussian, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 2)
    return binary

def extractTableLines(binary):
    # Extract vertical and horizontal lines from binary image"""
    verticalSE = cv.getStructuringElement(cv.MORPH_RECT, (1, 20))
    verticalLines = cv.erode(binary, verticalSE, iterations=5)
    verticalLines = cv.dilate(verticalLines, verticalSE, iterations=5)
    
    horizontalSE = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    horizontalLines = cv.erode(binary, horizontalSE, iterations=5)
    horizontalLines = cv.dilate(horizontalLines, horizontalSE, iterations=5)
    
    return verticalLines, horizontalLines

def getTableMask(verticalLines, horizontalLines):
    # Combine vertical and horizontal lines to create a table mask
    tableMask = cv.add(verticalLines, horizontalLines)
    tableMask = cv.erode(tableMask, cv.getStructuringElement(cv.MORPH_RECT, (2,2)), iterations = 2)
    tableMask = cv.dilate(tableMask, cv.getStructuringElement(cv.MORPH_RECT, (2,2)), iterations = 1)
    return tableMask

def find_largest_contour(tableMask):
    # Find the largest contour in the table mask
    contours, _ = cv.findContours(tableMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    table_contour = max(contours, key=cv.contourArea)
    return table_contour

def warp_table(paperImage, table_contour):
    # Warp the detected table to a top-down view
    rect = cv.minAreaRect(table_contour)
    box = cv.boxPoints(rect)
    box = np.array(box, dtype="float32")
    
    s = box.sum(axis=1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    ordered_box = np.array([tl, tr, br, bl], dtype="float32")
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([[0,0], [maxWidth,0], [maxWidth,maxHeight], [0,maxHeight]], dtype="float32")
    M = cv.getPerspectiveTransform(ordered_box, dst)
    table_image = cv.warpPerspective(paperImage, M, (maxWidth, maxHeight))
    
    return table_image

def extractTable(imagePath):
    imageRGB = readImage(imagePath)
    paperImage = getPaperImage(imagePath)
    binary = threshold(paperImage)
    verticalLines, horizontalLines = extractTableLines(binary)
    tableMask = getTableMask(verticalLines, horizontalLines)
    tableContour = find_largest_contour(tableMask)
    tableImage = warp_table(paperImage, tableContour)
    return tableImage

# Testing for all dataset
def testTableExtraction():
    images = ["../data/images/1.jpg", "../data/images/2.jpg", "../data/images/3.jpg", "../data/images/4.jpg", "../data/images/5.jpg", "../data/images/6.jpg", "../data/images/7.jpg", "../data/images/8.jpg", "../data/images/9.jpg",
               "../data/images/10.jpg", "../data/images/11.jpg", "../data/images/12.jpg", "../data/images/13.jpg", "../data/images/14.jpg", "../data/images/15.jpg", "../data/images/16.jpg", "../data/images/17.jpg", "../data/images/18.jpg",
                "../data/images/19.jpg", "../data/images/20.jpg", "../data/images/21.jpg", "../data/images/22.jpg", "../data/images/23.jpg","../data/images/24.jpg"]
    for i, imagePath in enumerate(images):
        originalImage = readImage(imagePath)
        tableImage = extractTable(imagePath)
        show_images([originalImage, tableImage], titles = [f"Original {i + 1}", f"Scanned Table {i + 1}"])
        outputPath = f"../data/tables/{i + 1}.jpg"
        cv.imwrite(outputPath, tableImage)
        print(f"Saved: {outputPath}")

if __name__ == "__main__":
    testTableExtraction()

