import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
import os

# 1) Image reading and preprocessing
def readImage(path):
    # Read an image and convert to RGB.
    imageBGR = cv.imread(path)
    imageRGB = cv.cvtColor(imageBGR, cv.COLOR_BGR2RGB)
    return imageRGB

def convertToGrayScale(imageRGB):
    # Convert RGB image to grayscale.
    return cv.cvtColor(imageRGB, cv.COLOR_BGR2GRAY)

def applyGaussian(imageGrayScale, windowSize = (7,7)):
    # Apply Gaussian blur to reduce noise.
    return cv.GaussianBlur(imageGrayScale, windowSize, 0)

def thresholdImage(imageGrayScale):
    # Return a binary image using adaptive threshold.
    binary = cv.adaptiveThreshold(imageGrayScale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    return binary

# 2) Contour detection and paper extraction
def findBiggestFourSidedContour(thresh_image):
    # Find the largest 4-sided contour in a binary image.
    contours, _ = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        constructedPoly = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(constructedPoly) == 4 and area > maxArea:
            maxArea = area
            biggestContour = constructedPoly
    return biggestContour

def segmentPaper(imageRGB, contour):
    # Create a masked image containing only the paper.
    gray = cv.cvtColor(imageRGB, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv.fillPoly(mask, [contour], 255)
    extractedPaper = cv.bitwise_and(imageRGB, imageRGB, mask)
    return extractedPaper

# 3) Perspective warp
def orderPoints(contourPoints):
    # Order points: top-left, top-right, bottom-left, bottom-right
    contourPoints = contourPoints.reshape(4, 2)
    contourPoints = sorted(contourPoints, key=lambda p: p[1])  # sort by y
    top = contourPoints[:2]     # top two points (smallest y)
    bottom = contourPoints[2:]  # bottom two points (largest y)
    # Getting Top Left and Top Right
    if top[0][0] < top[1][0]:
        topLeft, topRight = top[0], top[1]
    else:
        topLeft, topRight = top[1], top[0]
    # Getting Bottom Left and Bottom Right
    if bottom[0][0] < bottom[1][0]:
        bottomLeft, bottomRight = bottom[0], bottom[1]
    else:
        bottomLeft, bottomRight = bottom[1], bottom[0]
    # Final ordered points
    orderedPoints = [topLeft, topRight, bottomLeft, bottomRight]
    return np.float32(orderedPoints)

def calculateCroppedImageDimensions(orderedPoints):
    # Compute width and height of the rectangle from ordered points.
    topLeft, topRight, bottomLeft, bottomRight = orderedPoints[0], orderedPoints[1], orderedPoints[2], orderedPoints[3]
    leftEdge = (((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2)) ** (1/2)
    rightEdge = (((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2)) ** 0.5
    topEdge = (((topLeft[0] - topRight[0]) ** 2) + ((topLeft[1] - topRight[1]) ** 2)) ** 0.5
    bottomEdge = (((bottomLeft[0] - bottomRight[0]) ** 2) + ((bottomLeft[1] - bottomRight[1]) ** 2)) ** 0.5
    width = int(max(topEdge, bottomEdge))
    height = int(max(leftEdge, rightEdge))
    return width, height

def warpPaper(image, contour):
    # Warp the paper to a top-down view rectangle.
    orderedPoints = orderPoints(contour)
    width, height = calculateCroppedImageDimensions(orderedPoints)
    destinationPoints = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv.getPerspectiveTransform(orderedPoints, destinationPoints)
    warpedImage = cv.warpPerspective(image, matrix, (width, height))
    return warpedImage

# 4) Visualization functions
def drawContours(image, contours, biggest=None):
    # Draw all contours in green, highlight biggest in thick green.
    img_copy = image.copy()
    cv.drawContours(img_copy, contours, -1, (0,255,0), 2)
    if biggest is not None:
        cv.drawContours(img_copy, [biggest], -1, (0,255,0), 10)
    return img_copy

# 5) Main pipeline
def extractPaper(imagePath):
    imageRGB = readImage(imagePath)
    imageGrayScale = convertToGrayScale(imageRGB)
    imageGrayScale = applyGaussian(imageGrayScale)
    thresh = thresholdImage(imageGrayScale)
    biggestContour = findBiggestFourSidedContour(thresh)
    extractedPaper = segmentPaper(imageRGB, biggestContour)
    warped = warpPaper(imageRGB, biggestContour)
    return warped

# 6) Testing for all dataset
def test():
    images = ["../data/images/1.jpg", "../data/images/2.jpg", "../data/images/3.jpg", "../data/images/4.jpg", "../data/images/5.jpg", "../data/images/6.jpg", "../data/images/7.jpg", "../data/images/8.jpg", "../data/images/9.jpg",
               "../data/images/10.jpg", "../data/images/11.jpg", "../data/images/12.jpg", "../data/images/13.jpg", "../data/images/14.jpg", "../data/images/15.jpg", "../data/images/16.jpg", "../data/images/17.jpg", "../data/images/18.jpg",
                "../data/images/19.jpg", "../data/images/20.jpg", "../data/images/21.jpg", "../data/images/22.jpg", "../data/images/23.jpg","../data/images/24.jpg"]
    for i, imagePath in enumerate(images):
        paper = extractPaper(imagePath)
        image = readImage(imagePath)
        # show_images([image, paper], titles = [f"Original {i + 1}", f"Scanned Paper {i + 1}"])
        outputPath = f"../data/papers/{i + 1}.jpg"
        cv.imwrite(outputPath, paper)
        print(f"Saved: {outputPath}")

if __name__ == "main":
    test()
