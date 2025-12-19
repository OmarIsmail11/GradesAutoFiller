import cv2 as cv
import os
import easyocr

reader = easyocr.Reader(['en'], gpu=False) 

def recognize_student_name(cell_img_path, lang_input='en'):
    """
    Recognizes the student name from the printed name column.
    Uses EasyOCR (Pretrained DNN) for high-accuracy alphanumeric reading.
    """
    if not os.path.exists(cell_img_path):
        return "Error: File not found"
    
    reader = easyocr.Reader([lang_input], gpu=False)
    results = reader.readtext(cell_img_path)

    name_string = " ".join([res[1] for res in results])

    return name_string.strip().upper()

sample_name_cell = r"Module1-GradesSheet\data\cells\1\row2_cell3.jpg"
name = recognize_student_name(sample_name_cell, 'en')
print(f"Detected Name: {name}")