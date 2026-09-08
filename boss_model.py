"""
Phase 5 — Prompt Optimization (Boss Model)

Llama 70B reads failing examples from the train split, analyses the failure
patterns, and rewrites the zero_shot prompt. The optimised prompt is then
evaluated on held-out val and test splits and tested for transfer across all
five models.

Dataset split (out of 100 SST-2 examples used in eval.py):
  train  : idx  0–59   (60 examples — boss model sees failures here)
  val    : idx 60–79   (20 examples — prompt development feedback)
  test   : idx 80–99   (20 examples — never touched, final transfer test)

Note on SageMaker: the guide lists deploying Llama 70B as a SageMaker
endpoint for dedicated throughput. At this evaluation scale (~30 calls total)
the on-demand Bedrock path costs <$0.01 and avoids $3–6/hr endpoint charges.
The logic here works identically with a SageMaker endpoint — swap the bedrock
client for a sagemaker-runtime client when dedicated throughput is needed.
"""

import boto3
import json
import pathlib
import pandas as pd
import sys
import time
import re
from datetime import datetime, timezone
from config import MODELS, AWS_REGION, PROMPT_TEMPLATES
from db import write_result
from metrics import put_invocation_metrics

ACCURACY_THRESHOLD = 0.90  # skip optimisation if val accuracy is at or above this

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
BOSS_MODEL_KEY = "llama3-70b"

# ── Helpers (same as eval.py) ─────────────────────────────────────────────

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

def extract_label(raw_output):
    text = raw_output.lower().strip()
    first_word = text.split()[0].strip(".,!?") if text else ""
    for label in ["positive", "negative"]:
        if label in text or first_word == label:
            return label
    return "unknown"

def eval_prompt(prompt_template, examples, model_key):
    results = []
    for i, ex in enumerate(examples):
        result = invoke_model(model_key, prompt_template.format(text=ex["text"]))
        label = extract_label(result["output"])
        cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)
        results.append({
            "correct": int(label == ex["true_label"]),
            "predicted_label": label,
            "raw_output": result["output"],
            "latency_ms": result["latency_ms"],
            "cost_usd": cost,
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })
        time.sleep(0.1)
    return results

# ── Step 1: Load existing results, create splits ──────────────────────────

print("Loading existing results...")
df = pd.read_csv("results/all_models_all_prompts.csv")
df["example_idx"] = df.groupby(["model", "prompt_variant"]).cumcount()

train_df = df[df["example_idx"] < 60]
val_df   = df[(df["example_idx"] >= 60) & (df["example_idx"] < 80)]
test_df  = df[df["example_idx"] >= 80]

# Reconstruct example lists from CSV (text + true_label are stored)
def to_examples(subset_df, model="llama3-8b", prompt="zero_shot"):
    rows = subset_df[(subset_df.model == model) & (subset_df.prompt_variant == prompt)]
    return [{"text": r["text"], "true_label": r["true_label"]}
            for _, r in rows.iterrows()]

val_examples  = to_examples(val_df)
test_examples = to_examples(test_df)

# Baseline accuracy on val / test from existing results
baseline_val_acc  = val_df[(val_df.model == "llama3-8b") & (val_df.prompt_variant == "zero_shot")]["correct"].mean()
baseline_test_acc = test_df[(test_df.model == "llama3-8b") & (test_df.prompt_variant == "zero_shot")]["correct"].mean()

print(f"Baseline llama3-8b zero_shot  val={baseline_val_acc:.0%}  test={baseline_test_acc:.0%}")

if baseline_val_acc >= ACCURACY_THRESHOLD:
    print(f"\nVal accuracy {baseline_val_acc:.0%} >= threshold {ACCURACY_THRESHOLD:.0%} — prompt is already good. Skipping optimisation.")
    sys.exit(0)

# ── Step 2: Collect train failures for boss model ─────────────────────────

fail_rows = train_df[
    (train_df.model == "llama3-8b") &
    (train_df.prompt_variant == "zero_shot") &
    (train_df.correct == 0)
].head(10)

failure_block = "\n".join(
    f'  Text: "{r["text"]}"\n'
    f'  True label: {r["true_label"]}\n'
    f'  Model predicted: {r["predicted_label"]}\n'
    f'  Raw output: "{r["raw_output"]}"\n'
    for _, r in fail_rows.iterrows()
)

current_prompt = PROMPT_TEMPLATES["zero_shot"]

# ── Step 3: Ask Llama 70B to rewrite the prompt ───────────────────────────

boss_prompt = f"""You are an expert prompt engineer specialising in text classification.

A zero-shot sentiment classification prompt is failing on some movie review examples from the Stanford SST-2 dataset. The reviews use nuanced or literary language.

Current prompt template ({{text}} is replaced with the review at runtime):
---
{current_prompt}
---

Examples where this prompt produced wrong predictions:
{failure_block}

Your task:
1. Identify the failure patterns (e.g. literary phrasing, hedging, implied sentiment).
2. Write an improved prompt template that fixes those patterns.

HARD RULES — you must follow all of these exactly:
- Keep {{text}} as the only placeholder.
- Keep the instruction to reply with ONE word only.
- The last line of the prompt must be exactly "Sentiment:" so the model's next token is the label.
- The only valid output labels are "positive" or "negative" — make this explicit.
- Do NOT add any other placeholders or variables.

Output ONLY the improved prompt template — no explanation, no markdown, no surrounding quotes.
"""

print(f"\nAsking {BOSS_MODEL_KEY} to rewrite the zero_shot prompt...")
boss_result = invoke_model(BOSS_MODEL_KEY, boss_prompt, max_tokens=400)
optimized_prompt = boss_result["output"].strip()

boss_cost = compute_cost(boss_result["input_tokens"], boss_result["output_tokens"], BOSS_MODEL_KEY)
print(f"Boss model cost: ${boss_cost:.5f}")
print(f"\n── Optimised prompt ──\n{optimized_prompt}\n")

# ── Step 4: Validate {text} placeholder exists ────────────────────────────

if "{text}" not in optimized_prompt:
    print("WARNING: boss model omitted {text} placeholder — injecting fallback suffix.")
    optimized_prompt += "\n\nText: {text}\nSentiment:"

# ── Step 5: Evaluate optimised prompt on val split (llama3-8b) ───────────

run_id = f"boss-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
print(f"Run ID: {run_id}")

print("\nEvaluating optimised prompt on val split (llama3-8b)...")
val_results = eval_prompt(optimized_prompt, val_examples, "llama3-8b")
opt_val_acc = sum(r["correct"] for r in val_results) / len(val_results)
print(f"  Val accuracy  — baseline={baseline_val_acc:.0%}  optimised={opt_val_acc:.0%}  delta={opt_val_acc - baseline_val_acc:+.0%}")

for i, (ex, r) in enumerate(zip(val_examples, val_results)):
    write_result(run_id, "llama3-8b", "sentiment_boss_val", "optimised", i, {
        "text": ex["text"],
        "true_label": ex["true_label"],
        "predicted_label": r["predicted_label"],
        "correct": r["correct"],
        "latency_ms": r["latency_ms"],
        "cost_usd": r["cost_usd"],
    })
    put_invocation_metrics("llama3-8b", "sentiment_boss_val", "optimised",
                           r["latency_ms"], r["cost_usd"],
                           r["input_tokens"], r["output_tokens"])

# ── Step 5b: Write-back to config.py if val improved ─────────────────────

wrote_back = False
if opt_val_acc > baseline_val_acc:
    cfg_path = pathlib.Path("config.py")
    cfg_text = cfg_path.read_text(encoding="utf-8")
    old_zero_shot = PROMPT_TEMPLATES["zero_shot"]
    if old_zero_shot in cfg_text:
        cfg_text = cfg_text.replace(old_zero_shot, optimized_prompt)
        cfg_path.write_text(cfg_text, encoding="utf-8")
        wrote_back = True
        print(f"\nconfig.py updated — zero_shot prompt replaced (val {baseline_val_acc:.0%} → {opt_val_acc:.0%})")
    else:
        print("\nWARNING: could not locate original zero_shot prompt in config.py — skipping write-back.")
else:
    print(f"\nNo val improvement ({opt_val_acc:.0%} vs baseline {baseline_val_acc:.0%}) — config.py unchanged.")

# ── Step 6: Transfer test — all 5 models on test split ───────────────────

print("\nTransfer test — all 5 models on test split (20 examples)...")
transfer_rows = []

for model_key in MODELS:
    print(f"  {model_key}...")
    t_results = eval_prompt(optimized_prompt, test_examples, model_key)
    opt_test_acc = sum(r["correct"] for r in t_results) / len(t_results)

    # Baseline test acc for this model (only available for llama3-8b and mistral-7b)
    if model_key in ("llama3-8b", "mistral-7b"):
        base = test_df[(test_df.model == model_key) & (test_df.prompt_variant == "zero_shot")]["correct"].mean()
        delta_str = f"{opt_test_acc - base:+.0%}"
    else:
        base = None
        delta_str = "n/a (new model)"

    print(f"    test acc={opt_test_acc:.0%}  baseline={f'{base:.0%}' if base is not None else 'n/a'}  delta={delta_str}")

    for i, (ex, r) in enumerate(zip(test_examples, t_results)):
        write_result(run_id, model_key, "sentiment_boss_test", "optimised", i, {
            "text": ex["text"],
            "true_label": ex["true_label"],
            "predicted_label": r["predicted_label"],
            "correct": r["correct"],
            "latency_ms": r["latency_ms"],
            "cost_usd": r["cost_usd"],
        })
        put_invocation_metrics(model_key, "sentiment_boss_test", "optimised",
                               r["latency_ms"], r["cost_usd"],
                               r["input_tokens"], r["output_tokens"])

    transfer_rows.append({
        "model": model_key,
        "test_acc_optimised": round(opt_test_acc, 3),
        "test_acc_baseline": round(base, 3) if base is not None else None,
        "delta": round(opt_test_acc - base, 3) if base is not None else None,
    })

# ── Step 7: Save summary ──────────────────────────────────────────────────

summary = {
    "run_id": run_id,
    "optimised_prompt": optimized_prompt,
    "boss_model": BOSS_MODEL_KEY,
    "boss_model_cost_usd": round(boss_cost, 6),
    "baseline_val_acc": round(baseline_val_acc, 3),
    "optimised_val_acc": round(opt_val_acc, 3),
    "val_delta": round(opt_val_acc - baseline_val_acc, 3),
    "config_updated": wrote_back,
    "transfer": transfer_rows,
}

with open("results/boss_model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(transfer_rows).to_csv("results/boss_model_transfer.csv", index=False)

print(f"\n── Final Report ──")
print(f"Val   : baseline={baseline_val_acc:.0%}  optimised={opt_val_acc:.0%}  delta={opt_val_acc - baseline_val_acc:+.0%}")
print(f"\nTransfer results (test split, 20 examples):")
print(f"{'Model':<16} {'Optimised':>10} {'Baseline':>10} {'Delta':>8}")
print("-" * 48)
for row in transfer_rows:
    base_str = f"{row['test_acc_baseline']:.0%}" if row["test_acc_baseline"] is not None else "  n/a"
    delta_str = f"{row['delta']:+.0%}" if row["delta"] is not None else "  n/a"
    print(f"{row['model']:<16} {row['test_acc_optimised']:>9.0%} {base_str:>10} {delta_str:>8}")

print(f"\nResults written to DynamoDB  run_id={run_id}")
print("Summary saved to results/boss_model_summary.json")
