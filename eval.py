"""SST-2 binary sentiment eval: models x prompt variants."""

import argparse
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset

from config import PROMPT_TEMPLATES
from harness.bedrock import ModelUnavailable, compute_cost, invoke_model
from harness.parsers import legacy_sentiment_label
from harness.persist import Recorder, add_common_args


def load_test_data(n=100):
    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    return [
        {"text": item["sentence"], "true_label": "positive" if item["label"] == 1 else "negative"}
        for item in dataset.select(range(n))
    ]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__), ["llama3-8b", "mistral-7b"], 100)
    args = parser.parse_args()
    out = args.out or "results/all_models_all_prompts.csv"
    recorder = Recorder(enabled=not args.no_persist)

    run_id = f"sentiment-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    examples = load_test_data(n=args.n)

    all_results = []
    for model_key in args.models:
        try:
            for prompt_name, template in PROMPT_TEMPLATES.items():
                print(f"\nRunning {model_key} | {prompt_name}")
                for i, ex in enumerate(examples):
                    result = invoke_model(model_key, template.format(text=ex["text"]))
                    label = legacy_sentiment_label(result["output"])
                    metrics = {
                        "text": ex["text"],
                        "true_label": ex["true_label"],
                        "predicted_label": label,
                        "raw_output": result["output"],
                        "correct": int(label == ex["true_label"]),
                        "latency_ms": result["latency_ms"],
                        "cost_usd": compute_cost(result["input_tokens"], result["output_tokens"], model_key),
                    }
                    recorder.record(run_id, model_key, "sentiment", prompt_name, i, metrics, result)
                    all_results.append({
                        "model": model_key, "prompt_variant": prompt_name, "example_id": i, **metrics,
                        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                        "stop_reason": result["stop_reason"],
                    })
                    if (i + 1) % 10 == 0:
                        print(f"  {i+1}/{len(examples)} done...")
        except ModelUnavailable as e:
            print(f"WARNING: {model_key} unavailable after retries; its results are INCOMPLETE (rows so far are kept): {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(out, index=False)

    print("\n=== Results by Model + Prompt Variant (point estimates; see analysis/rescore_and_ci.py for CIs) ===")
    summary = df.groupby(["model", "prompt_variant"]).agg(
        accuracy=("correct", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        unknown_rate=("predicted_label", lambda x: (x == "unknown").mean()),
    )
    print(summary.round(4).to_string())
    print(f"\nCSV: {out}" + ("" if args.no_persist else f" | DynamoDB run_id: {run_id}"))


if __name__ == "__main__":
    main()
