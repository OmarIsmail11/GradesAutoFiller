import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Module1GradesSheet.src.FileHandler.FileHandler import FileHandler

if __name__ == "__main__":
    # Test logic
    handler = FileHandler()
    
    # TODO: ADD SYMBOLS
    my_config = [
        {'name': 'Student ID', 'type': 'Number', 'len': 7},
        {'name': 'Arabic Name', 'type': 'Arabic Name'},
        {'name': 'English Name', 'type': 'English Name'},
        {'name': 'Score_Q1', 'type': 'Written Number', 'len': 1},
        {'name': 'Symbol1', 'type': 'Symbol'},
        {'name': 'Symbol2', 'type': 'Symbol'}
    ]

    handler.process_image_to_excel("C:/Users/youse/Downloads/grade sheet/grade sheet/1.jpg", my_config, "Student_Grades.xlsx")
    print("Application initialized successfully!")
