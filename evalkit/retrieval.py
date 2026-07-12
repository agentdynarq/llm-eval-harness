"""Retrieval metrics for RAG systems.

Text metrics (see metrics.py) score a generated answer. These score the step
before it: did the retriever put the right documents in front of the model?
Each per-query metric takes a ranked list of retrieved ids and the relevant
ids, and returns a score in [0, 1]. They are deterministic and need no model.

`relevant` may be:
- a set/list of ids (binary relevance), or
- a mapping id -> gain (graded relevance, used by nDCG).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple


def _relevant_set(relevant) -> set:
    if isinstance(relevant, Mapping):
        return {key for key, gain in relevant.items() if gain}
    return set(relevant)


def _gains(relevant) -> dict:
    if isinstance(relevant, Mapping):
        return {key: float(gain) for key, gain in relevant.items()}
    return {doc: 1.0 for doc in relevant}


def _topk(retrieved: Sequence, k: Optional[int]) -> list:
    return list(retrieved) if k is None else list(retrieved)[: max(0, k)]


def hit_rate(retrieved: Sequence, relevant, k: Optional[int] = None) -> float:
    """1.0 if at least one relevant document is in the top-k, else 0.0."""
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(doc in rel for doc in _topk(retrieved, k)) else 0.0


def precision_at_k(retrieved: Sequence, relevant, k: Optional[int] = None) -> float:
    """Fraction of the top-k results that are relevant."""
    top = _topk(retrieved, k)
    if not top:
        return 0.0
    rel = _relevant_set(relevant)
    return sum(1 for doc in top if doc in rel) / len(top)


def recall_at_k(retrieved: Sequence, relevant, k: Optional[int] = None) -> float:
    """Fraction of the relevant documents that appear in the top-k."""
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    found = {doc for doc in _topk(retrieved, k) if doc in rel}
    return len(found) / len(rel)


def reciprocal_rank(retrieved: Sequence, relevant, k: Optional[int] = None) -> float:
    """1 / rank of the first relevant document (0.0 if none in the top-k)."""
    rel = _relevant_set(relevant)
    for i, doc in enumerate(_topk(retrieved, k)):
        if doc in rel:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved: Sequence, relevant, k: Optional[int] = None) -> float:
    """Mean of precision@i taken at each relevant hit, over all relevant docs."""
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    hits = 0
    running = 0.0
    for i, doc in enumerate(_topk(retrieved, k)):
        if doc in rel:
            hits += 1
            running += hits / (i + 1)
    return running / len(rel)


def ndcg_at_k(retrieved: Sequence, relevant, k: Optional[int] = None) -> float:
    """Normalized discounted cumulative gain; supports graded relevance."""
    gains = _gains(relevant)
    if not any(gains.values()):
        return 0.0
    top = _topk(retrieved, k)
    dcg = sum(gains.get(doc, 0.0) / math.log2(i + 2) for i, doc in enumerate(top))
    ideal = sorted((g for g in gains.values() if g > 0), reverse=True)
    if k is not None:
        ideal = ideal[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


RETRIEVAL_METRICS = {
    "hit_rate": hit_rate,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "reciprocal_rank": reciprocal_rank,
    "average_precision": average_precision,
    "ndcg": ndcg_at_k,
}


def evaluate_retrieval(
    cases: Sequence[Tuple[Sequence, object]],
    k: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
) -> dict:
    """Average each metric over many queries.

    `cases` is an iterable of (retrieved, relevant) pairs. The mean of
    `reciprocal_rank` is MRR and the mean of `average_precision` is MAP, so a
    full RAG scorecard comes out of one call.
    """
    names = list(metrics) if metrics else list(RETRIEVAL_METRICS)
    cases = list(cases)
    if not cases:
        return {name: 0.0 for name in names}
    result = {}
    for name in names:
        fn = RETRIEVAL_METRICS[name]
        result[name] = sum(fn(retrieved, relevant, k) for retrieved, relevant in cases) / len(cases)
    return result
