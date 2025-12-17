# Main entry point for the backend API
# This file imports from main_grop.py (Groq-based implementation)
# To use GPT instead, change the import to main_gpt

from main_grop import app

__all__ = ["app"]
