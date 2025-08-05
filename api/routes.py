# api/routes.py
from fastapi import APIRouter, Form, UploadFile, File
from typing import List, Optional
from core.service import process_generation_request

router = APIRouter()

@router.post("/generate")
async def generate_response(
    prompt: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    file_data = []

    # Swagger sendet manchmal "" statt None → filtern
    if files:
        clean_files = [f for f in files if not isinstance(f, str)]
        for file in clean_files:
            content = await file.read()
            file_data.append({
                "filename": file.filename,
                "content": content
            })

    # Service aufrufen
    return process_generation_request(prompt, file_data)
