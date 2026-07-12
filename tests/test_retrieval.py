"""Tests for retrieval metrics. Run: python -m unittest discover -s tests -v"""
import unittest

from evalkit.retrieval import (
    average_precision,
    evaluate_retrieval,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# A fixed ranking used across tests: relevant docs sit at ranks 2 and 4.
RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


class TestHitRate(unittest.TestCase):
    def test_miss_in_top1_hit_in_top2(self):
        self.assertEqual(hit_rate(RANKED, RELEVANT, k=1), 0.0)
        self.assertEqual(hit_rate(RANKED, RELEVANT, k=2), 1.0)

    def test_no_relevant_is_zero(self):
        self.assertEqual(hit_rate(RANKED, set()), 0.0)


class TestPrecisionRecall(unittest.TestCase):
    def test_precision_at_k(self):
        self.assertEqual(precision_at_k(RANKED, RELEVANT, k=2), 0.5)
        self.assertAlmostEqual(precision_at_k(RANKED, RELEVANT, k=4), 0.5)

    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(RANKED, RELEVANT, k=2), 0.5)
        self.assertEqual(recall_at_k(RANKED, RELEVANT, k=4), 1.0)

    def test_empty_retrieval(self):
        self.assertEqual(precision_at_k([], RELEVANT, k=3), 0.0)
        self.assertEqual(recall_at_k([], RELEVANT, k=3), 0.0)


class TestRankAware(unittest.TestCase):
    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(RANKED, RELEVANT), 0.5)  # first hit at rank 2
        self.assertEqual(reciprocal_rank(["d2"] + RANKED, RELEVANT), 1.0)
        self.assertEqual(reciprocal_rank(["x", "y"], RELEVANT), 0.0)

    def test_average_precision(self):
        # hits at ranks 2 and 4: (1/2 + 2/4) / 2 = 0.5
        self.assertAlmostEqual(average_precision(RANKED, RELEVANT), 0.5)

    def test_average_precision_perfect(self):
        self.assertAlmostEqual(average_precision(["d2", "d4", "d1"], RELEVANT), 1.0)


class TestNdcg(unittest.TestCase):
    def test_perfect_order_is_one(self):
        self.assertAlmostEqual(ndcg_at_k(["d2", "d4"], RELEVANT), 1.0)

    def test_between_zero_and_one(self):
        score = ndcg_at_k(RANKED, RELEVANT)
        self.assertTrue(0.0 < score < 1.0)

    def test_graded_relevance(self):
        graded = {"d1": 3, "d2": 2, "d3": 1}
        # ideal order is d1, d2, d3 which is exactly the ranking -> 1.0
        self.assertAlmostEqual(ndcg_at_k(["d1", "d2", "d3"], graded), 1.0)
        # a worse order scores lower
        self.assertLess(ndcg_at_k(["d3", "d2", "d1"], graded), 1.0)


class TestAggregate(unittest.TestCase):
    def test_evaluate_retrieval_means(self):
        cases = [
            (["d2", "d1"], {"d2"}),   # rr = 1.0
            (["d1", "d3"], {"d3"}),   # rr = 0.5
        ]
        agg = evaluate_retrieval(cases, k=2)
        self.assertAlmostEqual(agg["reciprocal_rank"], 0.75)  # MRR
        self.assertIn("ndcg", agg)
        self.assertIn("average_precision", agg)  # MAP

    def test_empty_cases(self):
        self.assertEqual(evaluate_retrieval([], metrics=["hit_rate"]), {"hit_rate": 0.0})


if __name__ == "__main__":
    unittest.main()
