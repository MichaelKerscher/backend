from lib.clients import gemini_client
# from lib.clients import openai_client  # später z. B. aktivieren

CLIENTS = {
    "gemini": gemini_client,
    # "openai": openai_client,
}