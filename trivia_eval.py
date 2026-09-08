import boto3
import pandas as pd
import re
import time
from datetime import datetime, timezone
from datasets import load_dataset
from config import MODELS, AWS_REGION, TRIVIA_PROMPT_TEMPLATES
from db import write_result
from metrics import put_invocation_metrics

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def invoke_model(model_key, prompt, max_tokens=100):
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

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_answer(raw_output):
    """For CoT: pull 'Answer: X' from last line. Otherwise take first line."""
    for line in reversed(raw_output.strip().split("\n")):
        if line.lower().startswith("answer:"):
            return line[7:].strip()
    return raw_output.strip().split("\n")[0].strip()

def exact_match(prediction, aliases):
    pred = normalize(prediction)
    return any(normalize(a) == pred for a in aliases)

def contains_answer(prediction, aliases):
    pred = normalize(prediction)
    return any(normalize(a) in pred for a in aliases)

def load_trivia_data(n=100):
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    examples = []
    for item in dataset.select(range(n)):
        aliases = item["answer"]["aliases"] + [item["answer"]["value"]]
        examples.append({
            "question": item["question"],
            "aliases": list(set(aliases)),
        })
    return examples

if __name__ == "__main__":
    run_id = f"trivia-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    print("Loading TriviaQA dataset...")
    examples = load_trivia_data(n=100)

    all_results = []
    models_to_run = list(MODELS.keys())

    for model_key in models_to_run:
        for prompt_name, template in TRIVIA_PROMPT_TEMPLATES.items():
            print(f"\nRunning {model_key} | {prompt_name}")
            for i, ex in enumerate(examples):
                prompt = template.format(question=ex["question"])
                result = invoke_model(model_key, prompt)
                answer = extract_answer(result["output"])
                cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
                em = int(exact_match(answer, ex["aliases"]))
                ca = int(contains_answer(answer, ex["aliases"]))

                metrics = {
                    "question": ex["question"],
                    "predicted_answer": answer,
                    "raw_output": result["output"],
                    "exact_match": em,
                    "contains_answer": ca,
                    "latency_ms": result["latency_ms"],
                    "cost_usd": cost,
                }

                write_result(run_id, model_key, "trivia", prompt_name, i, metrics)
                put_invocation_metrics(
                    model_key, "trivia", prompt_name,
                    result["latency_ms"], cost,
                    result["input_tokens"], result["output_tokens"],
                )

                all_results.append({"model": model_key, "prompt_variant": prompt_name, **metrics})

                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/100 done...")

    df = pd.DataFrame(all_results)
    df.to_csv("results/trivia_results.csv", index=False)

    print("\n=== TriviaQA Results by Model + Prompt ===")
    summary = df.groupby(["model", "prompt_variant"]).agg(
        exact_match_rate=("exact_match", "mean"),
        contains_rate=("contains_answer", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    ).round(4)

    for (model, variant), row in summary.iterrows():
        print(f"\n{model} | {variant}")
        print(f"  Exact match   : {row['exact_match_rate']:.1%}")
        print(f"  Contains ans  : {row['contains_rate']:.1%}")
        print(f"  Avg latency   : {row['avg_latency_ms']:.0f} ms")
        print(f"  Total cost    : ${row['total_cost_usd']:.4f}")

    print(f"\nAll results written to DynamoDB under run_id: {run_id}")
