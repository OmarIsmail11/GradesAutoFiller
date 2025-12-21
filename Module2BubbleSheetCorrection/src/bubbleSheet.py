import cv2 as cv
import numpy as np
from answersExtraction import *
from idExtraction import *
from paperExtraction import *
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def readModelAnswer(filePath):
    modelAnswers = []
    with open(filePath, "r") as f:
        for line in f:
            answer = line.strip()
            if answer:
                modelAnswers.append(answer)
    return modelAnswers

def gradePapers(paperImages, modelAnswers):
    results = []

    max_grade = len(modelAnswers)
    outputExcel = "grades.xlsx"

    for img_path in paperImages:
        # Load paper
        if isinstance(img_path, str):
            paper = extractPaper(img_path)
        else:
            paper = img_path

        student_id = detectID(paper)
        student_answers = detectAllAnswers(paper)

        grades = []
        for student_ans, correct_ans in zip(student_answers, modelAnswers):
            if student_ans == 'Z':
                grade = 0
            elif student_ans == correct_ans:
                grade = 1
            elif student_ans == 'X':
                grade = -1
            else:
                grade = 0
            grades.append(grade)

        total_score = sum(grades)
        results.append([student_id] + grades + [total_score])

    # Prepare DataFrame
    col_names = ["ID"] + [f"Q{i+1}" for i in range(len(modelAnswers))] + [f"Total / {max_grade}"]
    df = pd.DataFrame(results, columns=col_names)

    # Save to Excel
    df.to_excel(outputExcel, index=False)

    # Load Excel for formatting
    wb = load_workbook(outputExcel)
    ws = wb.active

    yellow_fill = PatternFill(start_color="FFFFFF00", end_color="FFFFFF00", fill_type="solid")  # wrong
    red_fill = PatternFill(start_color="FFFF0000", end_color="FFFF0000", fill_type="solid")  # multiple

    # Apply highlights
    for row_idx in range(2, len(df)+2):  # skip header
        for col_idx in range(2, 2 + len(modelAnswers)):  # Q1..Qn
            cell_value = ws.cell(row=row_idx, column=col_idx).value
            if cell_value == -1:
                ws.cell(row=row_idx, column=col_idx).fill = red_fill
            elif cell_value == 0:
                ws.cell(row=row_idx, column=col_idx).fill = yellow_fill

    wb.save(outputExcel)
    print(f"Graded Excel with highlights saved to {outputExcel}")


# ----------------------------
# Example usage
# ----------------------------
filePath = r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\modelAnswer.txt"
modelAnswers = readModelAnswer(filePath)

paperPaths = [
    r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\Module2BubbleSheetCorrection\data\images\1.jpg",
    r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\Module2BubbleSheetCorrection\data\images\2.jpg",
    r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\Module2BubbleSheetCorrection\data\images\3.jpg",
    r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\Module2BubbleSheetCorrection\data\images\4.jpg",
    r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\Module2BubbleSheetCorrection\data\images\5.jpg"
]

gradePapers(paperPaths, modelAnswers)

