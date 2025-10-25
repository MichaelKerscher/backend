# api/routes.py
from fastapi import APIRouter, Form, UploadFile, File
from typing import List, Optional
import json
from core.service import process_generation_request
from core.rag_service import query_rag_engine

router = APIRouter()

@router.post("/generate")
async def generate_response(
    metadata: str = Form(...),  # JSON als String
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Erwartet multipart/form-data mit:
    - metadata: JSON-String wie aus deinem Testframework
    - files[]: 0..n Dateien (Bilder, Audio, Video)
    """

    # JSON parsen
    try:
        data = json.loads(metadata)
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON in metadata"}

    input_data = data.get("input", {})
    prompt = input_data.get("prompt") or input_data.get("prompts")

    # Dateien einlesen (direkt als Bytes)
    file_data = []
    if files:
        for file in files:
            content = await file.read()
            file_data.append({
                "filename": file.filename,
                "data": content
            })

    # An Service übergeben
    result = process_generation_request(
        prompt=prompt,
        files=file_data,
        context=input_data.get("context"),
        model=data.get("model", "gemini-2.5-flash"),
        client_name=data.get("client", "gemini")
    )

    return {
        "test_id": data.get("test_id"),
        "status": "success",
        "response": result
    }

@router.post("/rag_query")
async def rag_query(payload: dict):
    """
    Calls the Vertex AI RAG Engine and returns the grounded response.
    Example:
      {"query": "Wie wird bei einem Gasleck laut Notfallprozedur reagiert?"}
    """
    try:
        query = payload.get("query")
        if not query:
            return {"status": "error", "message": "Missing field 'query'"}

        response_text = query_rag_engine(query)
        return {
            "status": "success",
            "query": query,
            "response": response_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
