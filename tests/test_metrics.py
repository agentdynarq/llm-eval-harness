"""Tests for the metrics. Run: python -m unittest discover -s tests -v"""
import unittest
from evalkit.metrics import normalize, exact_match, contains, token_f1, rouge_l


class TestNormalize(unittest.TestCase):
    def test_lowercases_and_strips_articles_and_punctuation(self):
        self.assertEqual(normalize("The Paris!"), "paris")
        self.assertEqual(normalize("  A  Cat.  "), "cat")


class TestExactMatch(unittest.TestCase):
    def test_matches_after_normalization(self):
        self.assertEqual(exact_match("Paris", "the paris"), 1.0)
        self.assertEqual(exact_match("London", "Paris"), 0.0)


class TestContains(unittest.TestCase):
    def test_reference_inside_prediction(self):
        self.assertEqual(contains("The capital is Paris.", "Paris"), 1.0)
        self.assertEqual(contains("The capital is Paris.", "Berlin"), 0.0)


class TestTokenF1(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertEqual(token_f1("hello world", "hello world"), 1.0)

    def test_no_overlap_is_zero(self):
        self.assertEqual(token_f1("cat dog", "fish bird"), 0.0)

    def test_partial_overlap(self):
        score = token_f1("the quick brown fox", "the brown fox")
        self.assertTrue(0.0 < score < 1.0)


class TestRougeL(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertEqual(rouge_l("a b c d", "a b c d"), 1.0)

    def test_subsequence_scores_between(self):
        score = rouge_l("a x b y c", "a b c")
        self.assertTrue(0.0 < score <= 1.0)


if __name__ == "__main__":
    unittest.main()
