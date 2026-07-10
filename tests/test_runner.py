"""Tests for the evaluation runner and aggregation."""
import os
import unittest
from evalkit.runner import evaluate
from evalkit.dataset import load_jsonl


class TestRunner(unittest.TestCase):
    def test_aggregates_scores(self):
        items = [
            {"prediction": "Paris", "reference": "Paris"},       # exact match 1
            {"prediction": "London", "reference": "Paris"},      # exact match 0
        ]
        result = evaluate(items, ["exact_match"])
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["aggregate"]["exact_match"], 0.5)
        self.assertEqual(len(result["per_item"]), 2)

    def test_rejects_unknown_metric(self):
        with self.assertRaises(ValueError):
            evaluate([{"prediction": "a", "reference": "a"}], ["nonsense"])

    def test_sample_dataset_runs(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.jsonl")
        result = evaluate(load_jsonl(path))
        self.assertEqual(result["n"], 6)
        for value in result["aggregate"].values():
            self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
