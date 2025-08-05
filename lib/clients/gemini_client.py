# lib/clients/gemini_client.py

import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, Part

# .env laden
load_dotenv()

# Vertex AI aktivieren?
USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

# Authentifizierung: Dienstkonto-Credentials prüfen
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path or not os.path.isfile(credentials_path):
    raise EnvironmentError("Fehlende oder ungültige .json-Datei für GOOGLE_APPLICATION_CREDENTIALS.")

# Projektkonfiguration
project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION", "europe-west4")

if not project_id:
    raise EnvironmentError("GCP_PROJECT_ID ist nicht gesetzt. Bitte in der .env-Datei definieren.")

# Gemini-Client erstellen
client = genai.Client(
    http_options=HttpOptions(api_version="v1"),
    project=project_id,
    location=location
)

# Standardmodell
DEFAULT_MODEL = "gemini-2.5-flash"

# --------------------------------
# Hilfsfunktion: Kontext anhängen
# --------------------------------
def append_context_to_prompt(prompt: str, context: dict | None) -> str:
    """Ergänzt das Prompt um optionale Kontextinformationen."""
    if not context:
        return prompt

    parts = []
    # Standort
    if "location" in context and "lat" in context["location"] and "lon" in context["location"]:
        lat = context["location"]["lat"]
        lon = context["location"]["lon"]
        parts.append(f"Standort: ({lat}, {lon})")

    # Zeit
    if "timestamp" in context:
        parts.append(f"Zeit: {context['timestamp']}")

    # Gerät
    if "device" in context:
        parts.append(f"Gerät: {context['device']}")

    # Netzwerk
    if "network" in context:
        parts.append(f"Netzwerkstatus: {context['network']}")

    # Kontext anhängen
    if parts:
        prompt += "\n[Kontext: " + "; ".join(parts) + "]"

    return prompt


# --------------------------------
# Hauptfunktion
# --------------------------------
def generate(input_type: str, prompt: str, model: str = DEFAULT_MODEL, image_path: str = None, audio_path: str = None, video_path: str = None, context: dict = None) -> str:
    if input_type == "text":
        return generate_text(prompt, model, context)
    elif input_type == "image":
        return generate_image_understanding(prompt, image_path, model, context)
    elif input_type == "audio":
        return generate_audio_understanding(prompt, audio_path, model, context)
    elif input_type == "video":
        return generate_video_understanding(prompt, video_path, model, context)
    else:
        raise NotImplementedError(f"Input-Typ '{input_type}' ist in Gemini-Client noch nicht implementiert.")


# --------------------------------
# Text
# --------------------------------
def generate_text(prompt: str, model: str = DEFAULT_MODEL, context: dict = None) -> str:
    """Textantwort mit Gemini über Vertex AI erzeugen"""
    try:
        prompt_with_context = append_context_to_prompt(prompt, context)
        response = client.models.generate_content(
            model=model,
            contents=prompt_with_context
        )
        return response.text
    except Exception as e:
        return f"Fehler bei der Generierung: {e}"

# --------------------------------
# Bild
# --------------------------------
def generate_image_understanding(prompt: str, image_path: str, model: str = DEFAULT_MODEL, context: dict = None) -> str:
    if not image_path or not os.path.isfile(image_path):
        return f"Bildpfad ungültig oder nicht gefunden: {image_path}"

    try:
        image_part = Part.from_bytes(
            data=open(image_path, "rb").read(),
            mime_type="image/jpeg"
        )

        prompt_with_context = append_context_to_prompt(prompt, context)
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt_with_context,
                image_part
            ]
        )
        return response.text
    except Exception as e:
        return f"Fehler bei der Bildanalyse: {e}"

# --------------------------------
# Audio
# --------------------------------
def generate_audio_understanding(prompt: str, audio_path: str, model: str = DEFAULT_MODEL, context: dict = None) -> str:
    if not audio_path or not os.path.isfile(audio_path):
        return f"Audiodatei ungültig oder nicht gefunden: {audio_path}"
    
    try:
        with open(audio_path, "rb") as audio_file:
            audio_part = Part.from_bytes(
                data=audio_file.read(),
                mime_type="audio/wav" 
            )

        prompt_with_context = append_context_to_prompt(prompt, context)
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt_with_context,
                audio_part
            ]
        )
        return response.text
    except Exception as e:
        return f"Fehler bei der Audioanalyse: {e}"

# --------------------------------
# Video
# --------------------------------
def generate_video_understanding(prompt: str, video_path: str, model: str = DEFAULT_MODEL, context: dict = None) -> str:
    if not video_path or not os.path.isfile(video_path):
        return f"Videodatei ungültig oder nicht gefunden: {video_path}"

    try:
        with open(video_path, "rb") as video_file:
            video_part = Part.from_bytes(
                data=video_file.read(),
                mime_type="video/mp4"  
            )

        prompt_with_context = append_context_to_prompt(prompt, context)
        response = client.models.generate_content(
            model=model,
            contents=[
                prompt_with_context,
                video_part
            ]
        )
        return response.text
    except Exception as e:
        return f"Fehler bei der Videoanalyse: {e}"
    
