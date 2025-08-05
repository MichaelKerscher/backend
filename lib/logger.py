import os
import json
import ast
from datetime import datetime

def log_response(test_id, prompt, response_text, model, client, runtime_seconds=None, input_type="text", result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)

    # Prüfen, ob es sich um Multi-Prompt-Antworten (Liste) handelt
    if isinstance(response_text, list):
        responses = []
        for idx, r in enumerate(response_text, start=1):
            # r sollte ein dict sein, sonst ist es evtl. ein String
            if isinstance(r, dict):
                text_block = r.get("text", r)
            else:
                # Versuch, String in Dict zu konvertieren
                try:
                    text_block = ast.literal_eval(r)
                except:
                    text_block = {"prompt": "", "response": str(r)}

            # Textblock weiter verarbeiten
            prompt_txt = text_block.get("prompt", "")
            resp_txt = text_block.get("response", "")
            token_count = len(resp_txt.split())
            responses.append({
                "prompt_index": idx,
                "text": {
                    "prompt": prompt_txt,
                    "response": resp_txt,
                    "runtime_seconds": text_block.get("runtime_seconds", None)
                },
                "tokens": token_count
            })
        response_data = responses
    else:
        # Single-Prompt-Fall
        resp_text = str(response_text)
        response_data = {
            "text": resp_text,
            "tokens": len(resp_text.split())
        }

    log_data = {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "client": client,
        "input": {
            "type": input_type,
            "prompt": prompt
        },
        "response": response_data,
        "meta": {
            "status": "success",
            "runtime_seconds": runtime_seconds if runtime_seconds is not None else -1
        }
    }

    filename = f"{test_id}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
    filepath = os.path.join(result_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"[LOG] Ergebnis gespeichert unter: {filepath}")
