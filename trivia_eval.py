"""TriviaQA (rc.nocontext) open-domain QA eval: exact match + contains-answer."""

import argparse
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset

from config import MODELS, TRIVIA_PROMPT_TEMPLATES
from harness.bedrock import ModelUnavailable, compute_cost, invoke_model
from harness.parsers import contains_answer, exact_match, extract_trivia_answer
from harness.persist import Recorder, add_common_args


def load_trivia_data(n=100):
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    examples = []
    for item in dataset.select(range(n)):
        aliases = item["answer"]["aliases"] + [item["answer"]["value"]]
        examples.append({"question": item["question"], "aliases": list(set(aliases))})
    return examples


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__), list(MODELS), 100)
    args = parser.parse_args()
    out = args.out or "results/trivia_results.csv"
    recorder = Recorder(enabled=not args.no_persist)

    run_id = f"trivia-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    examples = load_trivia_data(n=args.n)

    all_results = []
    for model_key in args.models:
        try:
            for prompt_name, template in TRIVIA_PROMPT_TEMPLATES.items():
                print(f"\nRunning {model_key} | {prompt_name}")
                for i, ex in enumerate(examples):
                    result = invoke_model(model_key, template.format(question=ex["question"]), max_tokens=100)
                    answer = extract_trivia_answer(result["output"])
                    metrics = {
                        "question": ex["question"],
                        "predicted_answer": answer,
                        "raw_output": result["output"],
                        "exact_match": int(exact_match(answer, ex["aliases"])),
                        "contains_answer": int(contains_answer(answer, ex["aliases"])),
                        "latency_ms": result["latency_ms"],
                        "cost_usd": compute_cost(result["input_tokens"], result["output_tokens"], model_key),
                    }
                    recorder.record(run_id, model_key, "trivia", prompt_name, i, metrics, result)
                    all_results.append({
                        "model": model_key, "prompt_variant": prompt_name, "example_id": i, **metrics,
                        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                        "stop_reason": result["stop_reason"],
                    })
                    if (i + 1) % 20 == 0:
                        print(f"  {i+1}/{len(examples)} done...")
        except ModelUnavailable as e:
            print(f"WARNING: {model_key} unavailable after retries; its results are INCOMPLETE (rows so far are kept): {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(out, index=False)

    print("\n=== TriviaQA by Model + Prompt (point estimates; see analysis/rescore_and_ci.py for CIs) ===")
    summary = df.groupby(["model", "prompt_variant"]).agg(
        exact_match_rate=("exact_match", "mean"),
        contains_rate=("contains_answer", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    )
    print(summary.round(4).to_string())
    print(f"\nCSV: {out}" + ("" if args.no_persist else f" | DynamoDB run_id: {run_id}"))


if __name__ == "__main__":
    main()
