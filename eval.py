import boto3
import pandas as pd
import time
import uuid
from datetime import datetime, timezone
from datasets import load_dataset
from config import MODELS, AWS_REGION, PROMPT_TEMPLATES
from db import write_result, get_run_results
from metrics import put_invocation_metrics

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def invoke_model(model_key, prompt, max_tokens=50):
    model_id = MODELS[model_key]["model_id"]
    start = time.time()
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0}
    )
    latency = time.time() - start
    output = response["output"]["message"]["content"][0]["text"].strip()
    input_tokens = response["usage"]["inputTokens"]
    output_tokens = response["usage"]["outputTokens"]
    return {
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": round(latency * 1000),
    }

def compute_cost(input_tokens, output_tokens, model_key):
    model = MODELS[model_key]
    return (
        (input_tokens / 1000) * model["price_per_1k_input_tokens"] +
        (output_tokens / 1000) * model["price_per_1k_output_tokens"]
    )

def extract_label(raw_output):
    text = raw_output.lower().strip()
    first_word = text.split()[0].strip(".,!?") if text else ""
    for label in ["positive", "negative"]:
        if label in text or first_word == label:
            return label
    return "unknown"

def load_test_data(n=100):
    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    examples = []
    for item in dataset.select(range(n)):
        examples.append({
            "text": item["sentence"],
            "true_label": "positive" if item["label"] == 1 else "negative"
        })
    return examples

if __name__ == "__main__":
    run_id = f"sentiment-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    print("Loading dataset...")
    examples = load_test_data(n=100)

    all_results = []
    models_to_run = ["llama3-8b", "mistral-7b"]

    for model_key in models_to_run:
        for prompt_name, template in PROMPT_TEMPLATES.items():
            print(f"\nRunning {model_key} | {prompt_name}")
            for i, ex in enumerate(examples):
                prompt = template.format(text=ex["text"])
                result = invoke_model(model_key, prompt)
                label = extract_label(result["output"])
                cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
                correct = label == ex["true_label"]

                metrics = {
                    "text": ex["text"],
                    "true_label": ex["true_label"],
                    "predicted_label": label,
                    "raw_output": result["output"],
                    "correct": int(correct),
                    "latency_ms": result["latency_ms"],
                    "cost_usd": cost,
                }

                # Write to DynamoDB immediately
                write_result(run_id, model_key, "sentiment", prompt_name, i, metrics)

                # Push metrics to CloudWatch
                put_invocation_metrics(
                    model_key, "sentiment", prompt_name,
                    result["latency_ms"], cost,
                    result["input_tokens"], result["output_tokens"]
                )

                all_results.append({
                    "model": model_key,
                    "prompt_variant": prompt_name,
                    **metrics
                })

                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/100 done...")

    # Save CSV as backup
    df = pd.DataFrame(all_results)
    df.to_csv("results/all_models_all_prompts.csv", index=False)

    print("\n=== Results by Model + Prompt Variant ===")
    summary = df.groupby(["model", "prompt_variant"]).agg(
        accuracy=("correct", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        unknown_rate=("predicted_label", lambda x: (x == "unknown").mean())
    ).round(4)

    for (model, variant), row in summary.iterrows():
        print(f"\n{model} | {variant}")
        print(f"  Accuracy     : {row['accuracy']:.1%}")
        print(f"  Avg latency  : {row['avg_latency_ms']:.0f} ms")
        print(f"  Total cost   : ${row['total_cost_usd']:.4f}")
        print(f"  Unknown rate : {row['unknown_rate']:.1%}")

    print(f"\nAll results written to DynamoDB under run_id: {run_id}")