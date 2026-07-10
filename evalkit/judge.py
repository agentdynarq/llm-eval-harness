"""Optional LLM-as-judge metric.

When an OpenAI key is available, this asks a model to rate how well a prediction
matches the reference on a 0 to 1 scale. It is a separate, opt-in metric because
it costs money and is non-deterministic, unlike the ones in metrics.py.
"""
import os
import re


def llm_judge(question: str, prediction: str, reference: str, model: str = "gpt-4o-mini") -> float:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        "Rate from 0.0 to 1.0 how well the ANSWER matches the REFERENCE for the "
        "QUESTION. Reply with only the number.\n\n"
        f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {prediction}\nSCORE:"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    match = re.search(r"[0-1](?:\.\d+)?", text or "")
    return max(0.0, min(1.0, float(match.group()))) if match else 0.0
