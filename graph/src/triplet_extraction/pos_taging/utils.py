import re
from typing import List, Dict, Set

def clean_text(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[!?]+", "", text)
    return text.strip().lower()