"""Load a predictions file. Each line is one item with a prediction and a
reference answer, and optionally an id and the original question."""
import json
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prediction" not in row or "reference" not in row:
                raise ValueError(f"line {n}: each item needs 'prediction' and 'reference'")
            items.append(row)
    return items
