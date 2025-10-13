from dotenv import load_dotenv
import os
from openai import OpenAI


def init_gpt() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment or .env file.")
    client = OpenAI(api_key=api_key)
    return client