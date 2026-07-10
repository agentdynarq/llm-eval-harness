"""Evaluate a predictions file against reference answers.

    python cli.py --predictions data/predictions.jsonl
    python cli.py --predictions data/predictions.jsonl --metrics exact_match token_f1 --out report.json
"""
import argparse
from evalkit.dataset import load_jsonl
from evalkit.runner import evaluate
from evalkit.report import format_table, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM answers against references.")
    parser.add_argument("--predictions", "-p", default="data/predictions.jsonl")
    parser.add_argument("--metrics", "-m", nargs="+", default=None,
                        help="subset of: exact_match contains token_f1 rouge_l")
    parser.add_argument("--out", "-o", default=None, help="write the full result to a JSON file")
    args = parser.parse_args()

    items = load_jsonl(args.predictions)
    result = evaluate(items, args.metrics)

    print(format_table(result))
    if args.out:
        save_json(result, args.out)
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
