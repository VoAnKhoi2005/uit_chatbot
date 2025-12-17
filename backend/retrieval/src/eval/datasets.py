import json
from typing import List, Dict

def load_eval_set(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
