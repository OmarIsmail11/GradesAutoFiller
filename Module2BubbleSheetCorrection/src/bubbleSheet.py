import cv2 as cv
import numpy as np
from answersExtraction import *
from idExtraction import *
from paperExtraction import *
import pandas as pd

imagePath = r"D:\Omar\Image Processing & Computer Vision\GradesAutoFiller\Module2BubbleSheetCorrection\data\images\6.jpg"
paper = extractPaper(imagePath)
height, width, channels = paper.shape

# Split paper into ID and questions regions
# idImage = paper[int(0.12*height):height//3, :5*width//9, :]

# Extract ID and answers
# student_id = extract_id(idImage)
answers = detectAllAnswers(paper)

# # -------------------------------
# # 2) Model answers
# # -------------------------------
# model_answers = ['A', 'B', 'C', 'C', 'B', 'A', 'A', 'B', 'C', 'C', 'C', 'B', 'B']

# # -------------------------------
# # 3) Compare and encode
# # -------------------------------
# encoded_answers = []
# for s_ans, m_ans in zip(student_answers, model_answers):
#     if s_ans == 'Z':         # Empty
#         encoded_answers.append(0)
#     elif s_ans == 'X':       # Multiple
#         encoded_answers.append(-1)
#     elif s_ans == m_ans:     # Correct
#         encoded_answers.append(1)
#     else:                    # Wrong
#         encoded_answers.append(0)

# # -------------------------------
# # 4) Create DataFrame
# # -------------------------------
# columns = ["ID"] + [f"Q{i+1}" for i in range(len(model_answers))]
# data = [ [student_id] + encoded_answers ]
# df = pd.DataFrame(data, columns=columns)

# # -------------------------------
# # 5) Save to Excel
# # -------------------------------
# df.to_excel("student_answers.xlsx", index=False)
# print("Excel saved with ID and answers!")
