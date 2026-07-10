"""Turn an evaluation result into a readable table and a JSON file."""
import json
from typing import Dict


def format_table(result: Dict) -> str:
    lines = [f"Evaluated {result['n']} item(s).", "", "Aggregate scores:"]
    for metric, value in result["aggregate"].items():
        bar = "#" * int(round(value * 20))
        lines.append(f"  {metric:12s} {value:5.3f}  {bar}")
    worst = sorted(
        result["per_item"],
        key=lambda r: sum(r["scores"].values()) / max(len(r["scores"]), 1),
    )[:3]
    if worst and result["n"] > 3:
        lines += ["", "Lowest-scoring items:"]
        for r in worst:
            avg = sum(r["scores"].values()) / len(r["scores"])
            lines.append(f"  [{r['id']}] avg {avg:.2f}  {r.get('question') or ''}")
    return "\n".join(lines)


def save_json(result: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
