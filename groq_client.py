import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load biến môi trường từ file .env

# Backward-compatible env lookup
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY/GROK_API_KEY không được tìm thấy trong file .env!")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_grok_llm(prompt: str) -> str:
    """Deprecated alias kept for compatibility."""
    return call_groq_llm(prompt)


def call_groq_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=60)
    response.raise_for_status()  # Raise lỗi nếu bị 401, 403, 500...

    result = response.json()
    return result["choices"][0]["message"]["content"]
