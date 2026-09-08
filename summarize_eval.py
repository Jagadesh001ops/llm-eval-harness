import boto3
import pandas as pd
import time
import torch
from datetime import datetime, timezone
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODELS, AWS_REGION
from db import write_result
from metrics import put_invocation_metrics

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

print("Loading NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")
nli_model.eval()
print("NLI model ready.")

SUMMARIZE_PROMPT = (
    "Summarize the following news article in 2-3 sentences. "
    "Only include facts from the article.\n\n"
    "Article: {article}\n\nSummary:"
)

def invoke_model(model_key, prompt, max_tokens=150):
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
        "latency_ms": round(latency * 1000)
    }

def compute_cost(input_tokens, output_tokens, model_key):
    model = MODELS[model_key]
    return (
        (input_tokens / 1000) * model["price_per_1k_input_tokens"] +
        (output_tokens / 1000) * model["price_per_1k_output_tokens"]
    )

def compute_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return round(scores["rougeL"].fmeasure, 4)

def check_faithfulness(summary, article):
    sentences = [s.strip() for s in summary.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    supported = 0
    truncated_article = article[:800]

    for sentence in sentences:
        inputs = nli_tokenizer(
            truncated_article,
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        predicted = torch.argmax(logits, dim=1).item()
        if predicted == 1:
            supported += 1

    return round(supported / len(sentences), 4)

def load_summarization_data(n=20):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    examples = []
    for item in dataset.select(range(n)):
        examples.append({
            "article": item["article"][:2000],
            "reference_summary": item["highlights"]
        })
    return examples

if __name__ == "__main__":
    run_id = f"summarization-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    print("Loading dataset...")
    examples = load_summarization_data(n=20)

    all_results = []

    for model_key in ["llama3-8b", "mistral-7b"]:
        print(f"\nRunning {model_key}...")
        for i, ex in enumerate(examples):
            prompt = SUMMARIZE_PROMPT.format(article=ex["article"])
            result = invoke_model(model_key, prompt)
            rouge = compute_rouge(result["output"], ex["reference_summary"])
            faithfulness = check_faithfulness(result["output"], ex["article"])
            cost = compute_cost(result["input_tokens"], result["output_tokens"], model_key)

            metrics = {
                "summary": result["output"],
                "rouge_l": rouge,
                "faithfulness": faithfulness,
                "latency_ms": result["latency_ms"],
                "cost_usd": cost,
            }

            # Write to DynamoDB immediately
            write_result(run_id, model_key, "summarization", "default", i, metrics)

            # Push metrics to CloudWatch
            put_invocation_metrics(
                model_key, "summarization", "default",
                result["latency_ms"], cost,
                result["input_tokens"], result["output_tokens"]
            )

            all_results.append({"model": model_key, "article_id": i, **metrics})

            if (i + 1) % 5 == 0:
                print(f"  {i+1}/20 done...")

    df = pd.DataFrame(all_results)
    df.to_csv("results/summarization_results.csv", index=False)

    print("\n=== Summarization Results by Model ===")
    summary = df.groupby("model").agg(
        avg_rouge_l=("rouge_l", "mean"),
        avg_faithfulness=("faithfulness", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    ).round(4)

    for model, row in summary.iterrows():
        print(f"\n{model}")
        print(f"  Avg ROUGE-L      : {row['avg_rouge_l']:.3f}")
        print(f"  Avg Faithfulness : {row['avg_faithfulness']:.1%}")
        print(f"  Avg Latency      : {row['avg_latency_ms']:.0f} ms")
        print(f"  Total Cost       : ${row['total_cost_usd']:.4f}")

    print(f"\nAll results written to DynamoDB under run_id: {run_id}")