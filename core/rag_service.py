# core/rag_service.py
import os
import time
import json
import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel

# --- Initialization ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "europe-west3")
CORPUS_NAME = os.getenv("VERTEX_RAG_CORPUS")

vertexai.init(project=PROJECT_ID, location=LOCATION)

def query_rag_engine(user_query: str) -> str:
    """Query Vertex AI RAG Engine using automatic retrieval configuration (SDK ≥ 1.71)."""

    start_time = time.perf_counter()

    # Step 1: Retrieve context chunks from the RAG corpus
    retrieval_response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=CORPUS_NAME)],
        text=user_query
    )

    # Concatenate retrieved context chunks
    contexts = []
    for i, ctx in enumerate(retrieval_response.contexts.contexts, start=1):
        contexts.append({
            "rank": i,
            "source_uri": getattr(ctx, "source_uri", "unknown"),
            "snippet": ctx.text[:400]
        })
    
    # Step 2: Combine context for prompt    
    context_text = "\n\n".join([c["snippet"] for c in contexts])

    # Combine context + user question for grounded generation
    prompt = f"{context_text}\n\nFrage: {user_query}"

    # Step 3: Generate grounded response
    model = GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    response_text = response.text.strip() if response and response.text else ""

    end_time = time.perf_counter()
    latency = round(end_time - start_time, 3)

    # Step 4: Logging to JSON file
    os.makedirs("results", exist_ok=True)
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": user_query,
        "latency_seconds": latency,
        "contexts": contexts,
        "response": response_text
    }

    # save to file
    safe_filename = user_query[:40].replace(" ", "_").replace("?", "")  # shorten
    log_path = f"results/rag_trace_{safe_filename}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)

    print(f"[LOG] RAG query logged to {log_path}")

    # Step 5: Return the response for the API
    return response_text
