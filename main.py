# main.py 
# FastAPI-Einstieg

from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional

app = FastAPI(title="Gemini Support Backend", version="0.1.0")

@app.post("/generate")
async def generate_response(
    prompt: str = Form(...),
    files: Optional[List[UploadFile]] = None
): 
    # 1. Dateien einlesen (falls vorhanden)
    file_data = []
    if files:
        for file in files:
            content = await file.read()
            file_data.append({"filename": file.filename, "data": content})

    # 2. Fake-Response für Start
    response = {
        "prompt": prompt,
        "num_files": len(file_data),
        "status": "processed"
    }
    return response