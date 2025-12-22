import cv2 as cv
import numpy as np
import math
import os
from ultralytics import YOLO

class ClassicalSymbolRecognizer:
    def __init__(self):
        pass

    def hasQuestionMark(self, cellImage):
        minCircleRadius = 3
        maxCircleRadius = 10
        minLineLength = 10
        grayScale = cv.cvtColor(cellImage, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(grayScale, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

        # Detect small circles (the dot)
        circles = cv.HoughCircles(binary, cv.HOUGH_GRADIENT, dp=1, minDist=5, param1=50, param2=10,
                                  minRadius=minCircleRadius, maxRadius=maxCircleRadius)
        if circles is None:
            return False

        # Detect vertical line by applying opening
        verticalLinesSE = cv.getStructuringElement(cv.MORPH_RECT, (1, minLineLength))
        verticalLines = cv.morphologyEx(binary, cv.MORPH_OPEN, verticalLinesSE, iterations=1)
        contours, _ = cv.findContours(verticalLines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False

        return True

    def hasSquare(self, cellImage):
        minArea = 20
        aspectRatioTolerance = 0.4
        grayScale = cv.cvtColor(cellImage, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(grayScale, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        SE = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, SE, iterations=1)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            area = cv.contourArea(contour)
            if len(approx) == 4 and area > minArea:
                x, y, width, height = cv.boundingRect(approx)
                aspectRatio = width / float(height)
                if (1 - aspectRatioTolerance) <= aspectRatio <= (1 + aspectRatioTolerance):
                    return True
        return False

    def hasTick(self, cellImage):
        grayScale = cv.cvtColor(cellImage, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(grayScale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

        lines = cv.HoughLinesP(thresh, 1, np.pi/180, threshold=15, minLineLength=60, maxLineGap=5)
        if lines is None:
            return False

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dx = x2 - x1
            dy = y1 - y2
            angle = math.degrees(math.atan2(dy, dx))
            if 15 <= angle <= 62:
                return True
        return False

    def detectHorizontalLines(self, cellImage):
        minLineLength = 60
        gray = cv.cvtColor(cellImage, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        horizontalSE = cv.getStructuringElement(cv.MORPH_RECT, (minLineLength, 1))
        horizontalLines = cv.morphologyEx(binary, cv.MORPH_OPEN, horizontalSE, iterations = 1)
        contours, _ = cv.findContours(horizontalLines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return len(contours)

    def detectVerticalLines(self, cellImage):
        minLineLength = 60
        grayScale = cv.cvtColor(cellImage, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(grayScale, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        verticalSE = cv.getStructuringElement(cv.MORPH_RECT, (1, minLineLength))
        verticalLines = cv.morphologyEx(binary, cv.MORPH_OPEN, verticalSE, iterations = 1)
        contours, _ = cv.findContours(verticalLines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return len(contours)

    def classifyCell(self, cellImage):
        if self.hasQuestionMark(cellImage):
            return "Question mark", 1
        if self.hasSquare(cellImage):
            return "Square", 1
        if self.hasTick(cellImage):
            return "Tick", 1

        vCount = self.detectVerticalLines(cellImage)
        if vCount > 0:
            return "Vertical lines", vCount

        hCount = self.detectHorizontalLines(cellImage)
        if hCount > 0:
            return "Horizontal lines", hCount

        return "Empty", 0

    def predict(self, cellImage):
        """Unified entry point matching the interface pattern."""
        label, count = self.classifyCell(cellImage)
        
        # Convert classification string to final value using your mapping logic
        if label == "Tick":
            return 5
        elif label == "Square":
            return 0
        elif label == "Empty":
            return ""
        elif label == "Question mark":
            return -1
        elif label == "Vertical lines":
            return count
        elif label == "Horizontal lines":
            return 5 - count
        else:
            return 0