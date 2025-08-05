# core/service.py

# ruft gemini_client + Logger

import os
import time
import tempfile
from typing import List, Optional
from lib.test_runner import run_test_from_file

def process_generation_request(
        prompt: str,
        files: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        model: str = "gemini-2.5-flash",
        client_name: str = "gemini"
) -> dict:
    """
    Nimmt Prompt und optionale Dateien entgegen,
    erstellt eine temporäre Testdatei im JSON-Format
    und führt dann das Framework aus.
    """

    import json
    from datetime import datetime

    # 1. Test-ID generieren
    test_id = f"api_{int(time.time())}"

    # 2. Eingabedaten vorbereiten
    input_data = {
        "type": "text",
        "prompt": prompt,
        "context": context or {}
    }

    # 3. Dateien temporär speichern, wenn vorhanden
    if files:
        # Aktuell behandeln wir nur 1 Datei pro Typ
        for f in files:
            filename = f["filename"]
            data = f["data"]

            # temporäre Datei schreiben
            tmp_path = os.path.join(tempfile.gettempdir(), filename)
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(data)

            # Heuristik: MIME Type erkennen über Endung
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_data["type"] = "image"
                input_data["image_path"] = tmp_path
            elif filename.lower().endswith((".wav", ".mp3")):
                input_data["type"] = "audio"
                input_data["audio_path"] = tmp_path
            elif filename.lower().endswith(".mp4"):
                input_data["type"] = "video"
                input_data["video_path"] = tmp_path
    
    # 4. Temporäre JSON-Testdatei erstellen
    tmp_json = os.path.join(tempfile.gettempdir(), f"{test_id}.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump({
            "test_id": test_id,
            "client": client_name,
            "model": model,
            "input": input_data
        }, f, ensure_ascii=False, indent=2)

    # 5. Test ausführen
    run_test_from_file(tmp_json)

    # 6. Logfile aus results/ holen (letztes File)
    results_dir = "results"
    log_files = sorted(
        [os.path.join(results_dir, f) for f in os.listdir(results_dir)],
        key=os.path.getmtime,
        reverse=True
    )
    latest_log = log_files[0] if log_files else None

    # 7. Ergebnis laden und zurückgeben
    if latest_log:
        with open(latest_log, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"status": "error", "message": "Kein Ergebnis gefunden"}
    

