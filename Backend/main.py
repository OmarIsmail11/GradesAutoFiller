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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Module1GradesSheet.src.FileHandler.FileHandler import fileHandler
from Module2BubbleSheetCorrection.src.bubbleSheet import gradePapers 


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

@app.post("/process-sheet")
async def process_sheet(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    columns: str = Form(...)
):
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

        fileHandler.process_image_to_excel(temp_path, column_config, output_filename)

        background_tasks.add_task(remove_file, temp_path)
        background_tasks.add_task(remove_file, output_filename)

        return FileResponse(
            path=output_filename,
            filename="Grading_Results.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade-bubbles")
async def grade_bubbles(
    background_tasks: BackgroundTasks,
    model_answer: UploadFile = File(...),
    paper_images: List[UploadFile] = File(...)
):
    session_id = uuid.uuid4().hex
    temp_dir = os.path.join("temp_bubbles", session_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        content = await model_answer.read()
        model_answers_list = [
            line.strip() for line in content.decode("utf-8").splitlines() if line.strip()
        ]

        image_paths = []
        for img in paper_images:
            file_path = os.path.join(temp_dir, img.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(img.file, f)
            image_paths.append(file_path)

        output_excel_path = os.path.join(temp_dir, "Bubble_Grades.xlsx")

        generated_file_path = gradePapers(image_paths, model_answers_list, output_excel_path)

        if not generated_file_path:
             raise HTTPException(status_code=500, detail="Grading function returned None.")

        background_tasks.add_task(remove_file, temp_dir)
        
        return FileResponse(
            path=generated_file_path,
            filename="Bubble_Sheet_Grades.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))