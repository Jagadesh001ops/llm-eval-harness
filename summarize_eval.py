"""CNN/DailyMail summarization eval: ROUGE-L + NLI faithfulness."""

import argparse
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer

from harness.bedrock import ModelUnavailable, compute_cost, invoke_model
from harness.persist import Recorder, add_common_args

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

SUMMARIZE_PROMPT = (
    "Summarize the following news article in 2-3 sentences. "
    "Only include facts from the article.\n\n"
    "Article: {article}\n\nSummary:"
)

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_nli = None


def _load_nli():
    global _nli
    if _nli is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        model.eval()
        _nli = (tok, model)
    return _nli


def compute_rouge(prediction, reference):
    return round(_scorer.score(reference, prediction)["rougeL"].fmeasure, 4)


def check_faithfulness(summary, article):
    """Fraction of summary sentences the NLI model labels as entailed by the article.

    KNOWN FLAW (kept so stored results stay reproducible; to be fixed in a later commit): the premise is
    truncated to the first 800 characters while the model summarised up to 2000, so sentences drawn from
    later in the article cannot be entailed and score as unfaithful.
    """
    import torch
    tok, model = _load_nli()
    sentences = [s.strip() for s in summary.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    supported = 0
    truncated_article = article[:800]
    for sentence in sentences:
        inputs = tok(truncated_article, sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        if torch.argmax(logits, dim=1).item() == 1:  # index 1 = entailment for this checkpoint
            supported += 1
    return round(supported / len(sentences), 4)


def load_summarization_data(n=20):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    return [{"article": item["article"][:2000], "reference_summary": item["highlights"]}
            for item in dataset.select(range(n))]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__), ["llama3-8b", "mistral-7b"], 20)
    args = parser.parse_args()
    out = args.out or "results/summarization_results.csv"
    recorder = Recorder(enabled=not args.no_persist)

    run_id = f"summarization-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Run ID: {run_id}")
    examples = load_summarization_data(n=args.n)

    all_results = []
    for model_key in args.models:
        print(f"\nRunning {model_key}...")
        try:
            for i, ex in enumerate(examples):
                result = invoke_model(model_key, SUMMARIZE_PROMPT.format(article=ex["article"]), max_tokens=150)
                metrics = {
                    "summary": result["output"],
                    "rouge_l": compute_rouge(result["output"], ex["reference_summary"]),
                    "faithfulness": check_faithfulness(result["output"], ex["article"]),
                    "latency_ms": result["latency_ms"],
                    "cost_usd": compute_cost(result["input_tokens"], result["output_tokens"], model_key),
                }
                recorder.record(run_id, model_key, "summarization", "default", i, metrics, result)
                all_results.append({
                    "model": model_key, "article_id": i, **metrics,
                    "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                    "stop_reason": result["stop_reason"],
                })
                if (i + 1) % 5 == 0:
                    print(f"  {i+1}/{len(examples)} done...")
        except ModelUnavailable as e:
            print(f"WARNING: {model_key} unavailable after retries; its results are INCOMPLETE (rows so far are kept): {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(out, index=False)

    print("\n=== Summarization by Model (point estimates; see analysis/rescore_and_ci.py for CIs) ===")
    summary = df.groupby("model").agg(
        avg_rouge_l=("rouge_l", "mean"),
        avg_faithfulness=("faithfulness", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    )
    print(summary.round(4).to_string())
    print(f"\nCSV: {out}" + ("" if args.no_persist else f" | DynamoDB run_id: {run_id}"))


if __name__ == "__main__":
    main()
