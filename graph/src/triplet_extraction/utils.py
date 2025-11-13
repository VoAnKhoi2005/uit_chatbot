import logging
import re
import csv


def clean_text(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)                # loại bỏ newline, tab
    text = re.sub(r"\s+", " ", text)                     # loại bỏ khoảng trắng thừa
    text = re.sub(r"[!?]+", "", text)                    # loại bỏ dấu !, ?
    text = text.replace('"', '')                         # loại bỏ tất cả dấu nháy kép
    text = re.sub(r"[^0-9a-zA-ZÀ-Ỹà-ỹđĐ\s\.\,\:\;\-\/]", " ", text)
    return text.strip().lower()


def load_synonym_dict(filepath):
    """Load synonym mappings from listSameKey.txt"""
    canonical_map = {}
    synonyms_map = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or "#" not in line:
                continue
            key, words_str = line.strip().split("#", 1)
            words = [w.strip().lower().replace("_", " ") for w in words_str.split(",") if w.strip()]

            if not words:
                continue

            # First word is the canonical name
            canonical_name = words[0]

            # Map all words (including canonical) to the canonical name
            for word in words:
                canonical_map[word] = canonical_name

            # Store all synonyms (excluding the canonical name itself)
            synonyms_map[canonical_name] = [w for w in words[1:] if w]

    return {'canonical': canonical_map, 'synonyms': synonyms_map}


def normalize_term(term, synonym_dict):
    """
    Normalize term by:
    1. Removing underscores and extra spaces
    2. Lowercasing
    3. Mapping to canonical name if exists in synonym dict
    """
    if not term:
        return None
    term = re.sub(r"_+", " ", term)  # Replace underscores with space
    term = re.sub(r"\s+", " ", term.strip())  # Clean multiple spaces
    term = term.lower()

    # Map to canonical name if exists
    if synonym_dict and 'canonical' in synonym_dict:
        return synonym_dict['canonical'].get(term, term)
    return term


def load_stopwords(stopword_file):
    """Load stopwords from CSV file"""
    stopwords = set()
    try:
        with open(stopword_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    stopwords.add(row[1].strip().lower())
    except Exception as e:
        print(f"Error loading stopwords: {e}")
    return stopwords


def is_valid_term(term, stopwords):
    """Check if term is not a stopword and not empty"""
    if not term or not term.strip():
        return False
    return term.lower() not in stopwords


import logging
import os


import logging
import os

def setup_logger(
    name="triplet_extraction",
    level=logging.INFO,
    log_to_file=False,
    file_path=None
):
    """
    Sets up a logger for .py files or notebooks with optional file and console output.

    Returns a tuple: (logger, console_handler, file_handler)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    # --- File handler (optional) ---
    file_handler = None
    if log_to_file:
        if not file_path:
            file_path = os.path.join(os.getcwd(), f"{name}.log")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {file_path}")

    return logger, console_handler, file_handler

