# lib/test_runner.py

import time
import json
from lib.logger import log_response
from lib.clients import CLIENTS
from lib.clients import gemini_client


"""Liest eine Testdatei (JSON), führt den Test aus und loggt das Ergebnis."""
def run_test_from_file(filepath: str):

    # Testdatei laden
    with open(filepath, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Basisdaten
    test_id = test_data["test_id"]
    client_name = test_data.get("client", "gemini").lower()
    model = test_data.get("model", "gemini-2.5-flash")

    input_data = test_data["input"]
    prompt = input_data.get("prompt")
    prompts = input_data.get("prompts")
    input_type = input_data.get("type", "text")
    image_path = input_data.get("image_path", None)
    audio_path = input_data.get("audio_path", None)
    video_path = input_data.get("video_path", None)
    context = input_data.get("context", None)

    # Client auswählen
    if client_name not in CLIENTS:
        raise ValueError(f"Unbekannter Client: '{client_name}'")
    client = CLIENTS[client_name]

    # ---------------------------------------
    # FALL 1: Mehrere Prompts
    # ---------------------------------------
    if prompts and isinstance(prompts, list):
        responses = []
        total_runtime = 0.0

        for idx, single_prompt in enumerate(prompts, start=1):
            print(f"[INFO] Starte Sub-Prompt {idx} für Test {test_id}...")
            start_time = time.perf_counter()
            response = client.generate(
                input_type=input_type,
                prompt=single_prompt,
                model=model,
                image_path=image_path,
                audio_path=audio_path,
                video_path=video_path,
                context=context
            )
            end_time = time.perf_counter()
            runtime = round(end_time - start_time, 3)
            total_runtime += runtime

            # Kontext anhängen
            prompt_with_context = gemini_client.append_context_to_prompt(single_prompt, context)

            responses.append({
                "prompt": prompt_with_context,
                "response": response,
                "runtime_seconds": runtime
            })

        log_response(
            test_id,
            prompts,
            responses,
            model,
            client_name,
            runtime_seconds=round(total_runtime, 3),
            input_type=input_type
        )
        return


    # ---------------------------------------
    # FALL 2: Einzel-Prompt (Standard)
    # ---------------------------------------

    # Ausführung + Messung
    start_time = time.perf_counter()
    response = client.generate(input_type=input_type, prompt=prompt, model=model, image_path=image_path, audio_path=audio_path, video_path=video_path, context=context)
    end_time = time.perf_counter()
    runtime = round(end_time - start_time, 3)

    # Ergebnis loggen
    prompt_with_context = gemini_client.append_context_to_prompt(prompt, context)
    log_response(test_id, prompt_with_context, response, model, client_name, runtime_seconds=runtime, input_type=input_type)
