import re

def clean_text(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[!?]+", "", text)
    return text.strip().lower()

def load_synonym_dict(filepath):
    """
    Load synonym mappings from listSameKey.txt.
    Format per line: A1#word1,word2,...
    Returns dict mapping each synonym -> canonical key, e.g. {"sống": "A1"}
    """
    synonym_map = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or "#" not in line:
                continue
            key, words_str = line.strip().split("#", 1)
            words = [w.strip().lower().replace("_", " ") for w in words_str.split(",") if w.strip()]
            for w in words:
                synonym_map[w] = key
    return synonym_map


def normalize_term(term, synonym_dict):
    """
    Normalize term by:
    1. Removing underscores and extra spaces
    2. Lowercasing
    3. Mapping to canonical synonym if exists
    """
    if not term:
        return None
    term = re.sub(r"_+", " ", term)          # Replace underscores with space
    term = re.sub(r"\s+", " ", term.strip()) # Clean multiple spaces
    term = term.lower()
    return synonym_dict.get(term, term)