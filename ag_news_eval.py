import boto3
import pandas as pd
import re
import time
from datetime import datetime, timezone
from datasets import load_dataset
from config import MODELS, AWS_REGION, AG_NEWS_PROMPT_TEMPLATES
from db import write_result
from metrics import put_invocation_metrics

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

AG_LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "science"}

LABEL_KEYWORDS = {
    "world":    ["world", "international", "global", "politics", "political", "foreign"],
    "sports":   ["sports", "sport", "athletic", "athletics", "game", "games", "soccer", "football", "basketball"],
    "business": ["business", "finance", "financial", "economy", "economic", "market", "stock", "trade"],
    "science":  ["science", "scientific", "technology", "tech", "sci", "research", "space", "computing"],
}

def invoke_model(model_key, prompt, max_tokens=50):
    model_id = MODELS[model_key]["model_id"]
    start = time.time()
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    latency = time.time() - start
    output = response["output"]["message"]["content"][0]["text"].strip()
    return {
        "output": output,
        "input_tokens": response["usage"]["inputTokens"],
        "output_tokens": response["usage"]["outputTokens"],
        "latency_ms": round(latency * 1000),
    }

def compute_cost(input_tokens, output_tokens, model_key):
    m = MODELS[model_key]
    return (input_tokens / 1000) * m["price_per_1k_input_tokens"] + \
           (output_tokens / 1000) * m["price_per_1k_output_tokens"]

def extract_ag_label(raw_output):
    text = raw_output.lower().strip()
    first_word = re.split(r'[\s.,!?:]', text)[0] if text else ""
    # Exact first-word match first
    for label, keywords in LABEL_KEYWORDS.items():
        if first_word in keywords:
            return label
    # Substring scan
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return label
    return "unknown"

def load_ag_news_data(n=100):
    dataset = load_dataset("fancyzhx/ag_news", split="test")
    examples = []
    for item in dataset.select(range(n)):
        examples.append({
            "text": item["text"],
            "true_label": AG_LABEL_MAP[item["label"]],
        })
    return examples

if __name__ == "__main__":
    run_id = f"ag-news-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    print("Loading AG News dataset...")
    examples = load_ag_news_data(n=100)

    all_results = []
    models_to_run = list(MODELS.keys())

    for model_key in models_to_run:
        for prompt_name, template in AG_NEWS_PROMPT_TEMPLATES.items():
            print(f"\nRunning {model_key} | {prompt_name}")
            for i, ex in enumerate(examples):
                prompt = template.format(text=ex["text"])
                result = invoke_model(model_key, prompt)
                label = extract_ag_label(result["output"])
                cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
                correct = int(label == ex["true_label"])

                metrics = {
                    "text": ex["text"][:500],
                    "true_label": ex["true_label"],
                    "predicted_label": label,
                    "raw_output": result["output"],
                    "correct": correct,
                    "latency_ms": result["latency_ms"],
                    "cost_usd": cost,
                }

                write_result(run_id, model_key, "ag_news", prompt_name, i, metrics)
                put_invocation_metrics(
                    model_key, "ag_news", prompt_name,
                    result["latency_ms"], cost,
                    result["input_tokens"], result["output_tokens"],
                )

                all_results.append({"model": model_key, "prompt_variant": prompt_name, **metrics})

                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/100 done...")

    df = pd.DataFrame(all_results)
    df.to_csv("results/ag_news_results.csv", index=False)

    print("\n=== AG News Results by Model + Prompt ===")
    summary = df.groupby(["model", "prompt_variant"]).agg(
        accuracy=("correct", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        unknown_rate=("predicted_label", lambda x: (x == "unknown").mean()),
    ).round(4)

    for (model, variant), row in summary.iterrows():
        print(f"\n{model} | {variant}")
        print(f"  Accuracy     : {row['accuracy']:.1%}")
        print(f"  Avg latency  : {row['avg_latency_ms']:.0f} ms")
        print(f"  Total cost   : ${row['total_cost_usd']:.4f}")
        print(f"  Unknown rate : {row['unknown_rate']:.1%}")

    print(f"\nAll results written to DynamoDB under run_id: {run_id}")
