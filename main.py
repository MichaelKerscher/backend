# main.py 
# FastAPI-Einstieg

from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI(title="Gemini Support Backend", version="0.1.0")

# API-Router registrieren
app.include_router(api_router)

