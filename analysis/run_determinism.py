"""How much do repeated runs of the same config differ? (Justifies item-level bootstrap over re-runs.)

Reads every run stored in the DynamoDB table llm-eval-results (needs AWS credentials with dynamodb:Scan),
and for each (task, model, prompt) with 2+ full runs reports: number of runs, accuracy range across runs,
and the fraction of items whose raw output was identical in every run.

Output: results/run_determinism.csv.   Run:  python analysis/run_determinism.py
"""

import pathlib
import sys

import boto3
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import AWS_REGION  # noqa: E402

SCORE_COL = {"sentiment": "correct", "ag_news": "correct", "trivia": "exact_match", "summarization": "rouge_l"}


def scan_all(table):
    items, kw = [], {}
    while True:
        r = table.scan(**kw)
        items += r["Items"]
        if "LastEvaluatedKey" not in r:
            return items
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def main():
    table = boto3.resource("dynamodb", region_name=AWS_REGION).Table("llm-eval-results")
    d = pd.DataFrame(scan_all(table))
    d = d[~d.run_id.str.startswith(("lambda-", "boss-"))]  # Lambda uses toy data; boss runs are one-offs
    d = d[d.task.isin(SCORE_COL)]
    rows = []
    for (task, model, prompt), g in d.groupby(["task", "model", "prompt_variant"]):
        sizes = g.groupby("run_id").size()
        full_runs = sizes[sizes == sizes.max()].index
        if len(full_runs) < 2:
            continue
        g = g[g.run_id.isin(full_runs)]
        col = SCORE_COL[task]
        acc = g.assign(s=g[col].astype(float)).groupby("run_id").s.mean()
        out_col = "summary" if task == "summarization" else "raw_output"
        same = g.pivot_table(index="example_id", columns="run_id", values=out_col, aggfunc="first").nunique(axis=1) == 1
        rows.append({"task": task, "model": model, "prompt_variant": prompt, "n_runs": len(full_runs),
                     "n_items": int(sizes.max()), "score_metric": col, "score_min": acc.min(), "score_max": acc.max(),
                     "score_range": acc.max() - acc.min(), "frac_items_identical_output": same.mean()})
    out = pd.DataFrame(rows)
    out.to_csv("results/run_determinism.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
