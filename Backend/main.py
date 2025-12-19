import os
import sys
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Module1GradesSheet.src.FileHandler.FileHandler import fileHandler

app = FastAPI(title="AI Grade Sheet Processor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-sheet")
async def process_sheet(
    image: UploadFile = File(...),
    columns: str = Form(...)
):
    temp_path = f"temp_{image.filename}"
    output_filename = "processed_results.xlsx"

    try:
        try:
            column_config = json.loads(columns)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in columns field")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        fileHandler.process_image_to_excel(temp_path, column_config, output_filename)

        return FileResponse(
            path=output_filename,
            filename="Grading_Results.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
