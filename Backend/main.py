import os
import sys
import json
import shutil
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

# --- PATH SETUP ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- MODULE 1: TABLE GRADING IMPORT ---
from Module1GradesSheet.src.FileHandler.FileHandler import fileHandler

# --- MODULE 2: BUBBLE SHEET IMPORT ---
# NOTE: Adjust this import path to match where you saved your Bubble Sheet 'gradePapers' function
try:
    # Assuming you organized the previous bubble sheet code into a module
    # If it's just a script, ensure it's in the python path
    from Module2BubbleSheetCorrection.src.bubbleSheet import gradePapers 
except ImportError:
    print("Warning: Could not import 'gradePapers'. Bubble sheet endpoint will fail if called.")
    # Dummy function to prevent crash if module is missing during setup
    def gradePapers(images, answers, output_path="grades.xlsx"):
        raise NotImplementedError("Bubble Sheet module not found.")

app = FastAPI(title="AI Grade Sheet Processor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER: CLEANUP ---
def remove_file(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

# ==========================================
# ENDPOINT 1: TABLE GRADING (Module 1)
# ==========================================
@app.post("/process-sheet")
async def process_sheet(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    columns: str = Form(...)
):
    # Unique temp file to avoid collisions
    unique_id = uuid.uuid4().hex
    temp_path = f"temp_{unique_id}_{image.filename}"
    output_filename = f"processed_{unique_id}.xlsx"

    try:
        try:
            column_config = json.loads(columns)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in columns field")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process
        fileHandler.process_image_to_excel(temp_path, column_config, output_filename)

        # Cleanup after response
        background_tasks.add_task(remove_file, temp_path)
        background_tasks.add_task(remove_file, output_filename)

        return FileResponse(
            path=output_filename,
            filename="Grading_Results.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        # cleanup if error occurs before response
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 2: BUBBLE SHEET (Module 2)
# ==========================================
@app.post("/grade-bubbles")
async def grade_bubbles(
    background_tasks: BackgroundTasks,
    model_answer: UploadFile = File(...),
    paper_images: List[UploadFile] = File(...)
):
    # 1. Setup Unique Temp Directory
    session_id = uuid.uuid4().hex
    temp_dir = os.path.join("temp_bubbles", session_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 2. Process Model Answer Key
        content = await model_answer.read()
        model_answers_list = [
            line.strip() for line in content.decode("utf-8").splitlines() if line.strip()
        ]

        # 3. Save Student Images
        image_paths = []
        for img in paper_images:
            file_path = os.path.join(temp_dir, img.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(img.file, f)
            image_paths.append(file_path)

        # 4. Define Output Path
        # We save the excel INSIDE the temp folder to avoid race conditions
        output_excel_path = os.path.join(temp_dir, "Bubble_Grades.xlsx")

        # 5. Call Grading Logic
        # Now passing the specific output path, and expecting a return value
        generated_file_path = gradePapers(image_paths, model_answers_list, output_excel_path)

        # Sanity check: ensure the function actually returned the path
        if not generated_file_path:
             raise HTTPException(status_code=500, detail="Grading function returned None.")

        # 6. Return Response
        # We rely on BackgroundTasks to clean up the folder AFTER the file is sent
        background_tasks.add_task(remove_file, temp_dir)
        
        return FileResponse(
            path=generated_file_path,
            filename="Bubble_Sheet_Grades.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        # Cleanup immediately if there was an error
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))