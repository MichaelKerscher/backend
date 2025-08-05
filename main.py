# main.py 
# FastAPI-Einstieg

# main.py
from fastapi import FastAPI

app = FastAPI(title="Gemini Support Backend", version="0.1.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}
