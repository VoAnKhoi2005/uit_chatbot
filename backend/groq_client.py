import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()  # Load biến môi trường từ file .env

logger = logging.getLogger(__name__)

# Backward-compatible env lookup
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY/GROK_API_KEY không được tìm thấy trong file .env!")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_grok_llm(prompt: str) -> str:
    """Deprecated alias kept for compatibility."""
    return call_groq_llm(system_prompt="", user_prompt=prompt)


def call_groq_llm(system_prompt: str = "", user_prompt: str = "", prompt: str | None = None) -> str:
    """
    Call Groq API with OpenAI-compatible chat format.
    
    Args:
        system_prompt: System message content (optional)
        user_prompt: User message content (required if prompt is None)
        prompt: Legacy single prompt (deprecated, will be treated as user_prompt)
    
    Returns:
        Response text from the model
    """
    # Backward compatibility: if prompt is provided, use it as user_prompt
    if prompt is not None:
        user_prompt = prompt
    
    if not user_prompt:
        raise ValueError("user_prompt is required")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build messages array with system and user messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    data = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
    except requests.HTTPError as e:
        error_msg = f"Groq API error {response.status_code}: {response.text}"
        logger.error(error_msg)
        logger.error(f"Request payload: model={GROQ_MODEL}, messages_count={len(messages)}")
        raise requests.HTTPError(error_msg, response=response) from e
    except requests.RequestException as e:
        logger.error(f"Groq API request failed: {e}")
        raise

    try:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected Groq response format: {response.text}")
        raise ValueError(f"Failed to parse Groq response: {e}") from e
