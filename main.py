# main.py 
# FastAPI-Einstieg

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

app = FastAPI(title="Gemini Support Backend", version="0.1.0")

# CORS-Middleware aktivieren
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

# API-Router registrieren
app.include_router(api_router)

