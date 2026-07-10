"""Apply the metrics to every item and aggregate the scores."""
from typing import List, Dict, Optional
from .metrics import METRICS


def evaluate(items: List[Dict], metric_names: Optional[List[str]] = None) -> Dict:
    metric_names = metric_names or list(METRICS)
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {unknown}. Available: {list(METRICS)}")

    per_item = []
    for i, item in enumerate(items):
        scores = {m: METRICS[m](item["prediction"], item["reference"]) for m in metric_names}
        per_item.append(
            {
                "id": item.get("id", i),
                "question": item.get("question"),
                "scores": scores,
            }
        )

    n = len(per_item)
    aggregate = {
        m: (sum(r["scores"][m] for r in per_item) / n if n else 0.0) for m in metric_names
    }
    return {"n": n, "metrics": metric_names, "aggregate": aggregate, "per_item": per_item}
