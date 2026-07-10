"""A small, deterministic evaluation harness for language-model answers."""
from .metrics import METRICS, normalize
from .runner import evaluate

__all__ = ["METRICS", "normalize", "evaluate"]
