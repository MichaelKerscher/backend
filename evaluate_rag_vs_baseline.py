"""
evaluate_rag_vs_baseline.py
---------------------------------------
Author: Michael Kerscher
Project: Integrating Enterprise Knowledge into Cloud-Native AI Systems using Google Cloud
Purpose:
    Automated evaluation of Gemini 2.5 Flash with and without enterprise knowledge grounding.
    Compares baseline responses (no RAG) and RAG-based responses (Vertex AI RAG Engine)
    for a fixed set of enterprise-domain queries.

Usage:
    1. Start backend:
        uvicorn main:app --host 0.0.0.0 --port 8000
    2. Run this script:
        python evaluate_rag_vs_baseline.py
    3. Results saved under:
        /results/
            - Qx_comparison.json : Detailed answers
            - summary_rag_vs_baseline.csv : Aggregate summary

---------------------------------------
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
API_BASE = "http://127.0.0.1:8000"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Representative test queries based on current RAG corpus
QUERIES = [
    {
        "id": "Q1",
        "query": "Wie soll laut Notfallprozedur bei einem Gasleck vorgegangen werden?"
    },
    {
        "id": "Q2",
        "query": "Welche Schritte sind in der Wartungs-Checkliste für Pumpstation PS-100 enthalten?"
    },
    {
        "id": "Q3",
        "query": "Was beschreibt die Gerätespezifikation für den SmartMeter AQ-500?"
    },
    {
        "id": "Q4",
        "query": "Welche Fehlercodes sind in der Vorfallsliste 2025 aufgeführt?"
    },
    {
        "id": "Q5",
        "query": "Wie unterscheidet sich die Gasleck-SOP von einer typischen Wasserleck-Prozedur?"
    },
]

def run_query(endpoint: str, payload=None, files=None):
    url = f"{API_BASE}/{endpoint}"
    t0 = time.perf_counter()
    if files:
        r = requests.post(url, files=files)
    else:
        r = requests.post(url, json=payload)
    t1 = time.perf_counter()
    latency = round(t1 - t0, 3)
    try:
        data = r.json()
    except Exception:
        data = {"status": "error", "response": r.text}
    return data, latency


def main():
    print("\n=== Running Baseline vs RAG Evaluation ===\n")
    results = []

    for q in QUERIES:
        query_text = q["query"]
        print(f"\n--- {q['id']} ---")
        print(f"Query: {query_text}")

        # Baseline (non-RAG)
        baseline_data, baseline_latency = run_query(
            "generate",
            files=[
                ("metadata", (None, json.dumps({
                    "test_id": f"{q['id']}_baseline",
                    "input": {"prompt": query_text},
                    "model": "gemini-2.5-flash",
                    "client": "gemini"
                }), "application/json"))
            ]
        )
        baseline_response = baseline_data.get("response", "")

        # RAG (grounded)
        rag_data, rag_latency = run_query(
            "rag_query",
            {"query": query_text}
        )
        rag_response = rag_data.get("response", "")

        # Log both results to JSON
        log_entry = {
            "id": q["id"],
            "query": query_text,
            "baseline_response": baseline_response,
            "rag_response": rag_response,
            "baseline_latency": baseline_latency,
            "rag_latency": rag_latency,
            "timestamp": datetime.now().isoformat()
        }

        log_path = os.path.join(RESULT_DIR, f"{q['id']}_comparison.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Saved detailed comparison → {log_path}")

        results.append(log_entry)

    # Summary Table
    df = pd.DataFrame(results)
    df["latency_diff_s"] = df["rag_latency"] - df["baseline_latency"]
    csv_path = os.path.join(RESULT_DIR, "summary_rag_vs_baseline.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\nEvaluation complete!")
    print(f"Summary CSV: {csv_path}")
    print("→ Inspect individual results in /results/*.json\n")


if __name__ == "__main__":
    main()
