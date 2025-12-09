import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load biến môi trường từ file .env

GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY không được tìm thấy trong file .env!")

GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_grok_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    response = requests.post(GROK_API_URL, headers=headers, json=data)
    response.raise_for_status()  # Raise lỗi nếu bị 401, 403, 500...

    result = response.json()
    return result["choices"][0]["message"]["content"]
