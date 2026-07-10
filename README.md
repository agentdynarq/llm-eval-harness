# llm-eval-harness

A small, deterministic **evaluation harness** for language-model answers. You give it a file of predictions and reference answers, and it scores each one with several metrics and reports the aggregate. It is the piece that turns "the model seems fine" into a number you can track as you change prompts, models, or a RAG pipeline.

The core metrics need no model and no API, so evaluation is fast and reproducible. An optional LLM-as-judge is included for the cases where exact strings are too strict.

## Why evaluation matters

When you tune a prompt, fine-tune a model, or tweak retrieval, you need to know if it actually got better. Eyeballing a few answers does not scale and hides regressions. A harness runs the same test set every time and gives comparable scores, so improvements (and regressions) are visible.

## Metrics

| Metric | What it measures |
|---|---|
| `exact_match` | Prediction equals the reference after normalization (lowercase, drop punctuation and articles) |
| `contains` | The reference appears inside the prediction (good for short factual answers) |
| `token_f1` | Token overlap between prediction and reference (precision + recall) |
| `rouge_l` | F1 over the longest common subsequence of tokens (rewards correct ordering) |
| `llm_judge` | Optional: a model rates the match 0 to 1 (needs an OpenAI key) |

## Quickstart

```bash
# core metrics need nothing beyond the standard library
python cli.py --predictions data/predictions.jsonl

# choose metrics and save the full per-item report
python cli.py -p data/predictions.jsonl -m exact_match token_f1 rouge_l -o report.json
```

Example output:

```
Evaluated 6 item(s).

Aggregate scores:
  exact_match  0.333  #######
  contains     0.500  ##########
  token_f1     0.611  ############
  rouge_l      0.585  ############

Lowest-scoring items:
  [5] avg 0.00  What is 2 plus 2?
```

## The data format

Each line in the predictions file is one item. `prediction` and `reference` are required; `id` and `question` are optional and only used in the report.

```json
{"id": 1, "question": "Capital of France?", "prediction": "It is Paris.", "reference": "Paris"}
```

To evaluate your own model or RAG pipeline, generate its answers, write them next to the reference answers in this format, and run the harness.

## Optional: LLM-as-judge

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # or put it in .env
```

`evalkit/judge.py` asks a model to rate a prediction against the reference from 0 to 1. Use it when answers are correct but worded differently, where exact match and token overlap are unfairly harsh.

## Project structure

```
llm-eval-harness/
  cli.py                run an evaluation from the command line
  evalkit/
    metrics.py          exact_match, contains, token_f1, rouge_l
    dataset.py          load the predictions jsonl
    runner.py           apply metrics to every item and aggregate
    report.py           readable table + JSON export
    judge.py            optional LLM-as-judge
  data/predictions.jsonl  sample data
  tests/                metric and runner tests
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- Retrieval metrics (hit-rate, MRR) for RAG systems.
- Per-metric confidence intervals via bootstrap.
- Compare two runs side by side and flag regressions.

## License

MIT
